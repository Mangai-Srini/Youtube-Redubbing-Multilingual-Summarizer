"""
YouTube Redubbing Multilingual Summarizer with Q&A Chatbot
Main application combining video processing, translation, voice cloning, and Q&A
"""
import torch
import sys

# This forces torch to load models even if they use the older "pickle" format
original_load = torch.load

def safe_load_wrapper(*args, **kwargs):
    # If the caller didn't specify weights_only, force it to False
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = safe_load_wrapper

import streamlit as st
import os
import torch
import whisper
import streamlit as st
import yt_dlp
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from TTS.api import TTS
import chromadb
from chromadb.config import Settings
import soundfile as sf
from pydub import AudioSegment
import tempfile
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig

# --- FIX: Allow Coqui TTS Classes in PyTorch 2.6+ ---
torch.serialization.add_safe_globals([XttsConfig, XttsArgs, XttsAudioConfig])
# ----------------------------------------------------

class YouTubeProcessor:
    """Handles YouTube video downloading and audio extraction"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def download_video(self, url, output_path="downloads"):
        """
        Downloads audio from a YouTube URL using yt-dlp and returns a tuple (audio_path, metadata).
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # yt-dlp configuration
        ydl_opts = {
            'format': 'bestaudio/best',      # Download best available audio quality
            'postprocessors': [{
                'key': 'FFmpegExtractAudio', # Extract audio using FFmpeg
                'preferredcodec': 'mp3',     # Convert to mp3
                'preferredquality': '192',
            }],
            'outtmpl': os.path.join(output_path, '%(title)s.%(ext)s'), # Output filename template
            'quiet': True,                   # Less terminal spam
            'no_warnings': True,
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # 1. Extract info to get the metadata and title (without downloading yet)
                info = ydl.extract_info(url, download=False)

                # 2. Download the audio (postprocessors will convert to .mp3)
                ydl.download([url])

                # 3. Construct the expected filename (post-processed to .mp3)
                sanitized_title = ydl.prepare_filename(info)
                final_filename = os.path.splitext(sanitized_title)[0] + ".mp3"

                # Prepare a small metadata dict to return alongside the audio path
                metadata = {
                    'id': info.get('id'),
                    'title': info.get('title'),
                    'uploader': info.get('uploader'),
                    'duration': info.get('duration'),
                    'webpage_url': info.get('webpage_url'),
                }

                return final_filename, metadata

        except Exception as e:
            raise Exception(f"Error downloading video: {str(e)}")

class TranscriptionEngine:
    """Transcribes audio using Whisper"""
    
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)
    
    def transcribe(self, audio_path: str, language: str = None) -> Dict:
        """Transcribe audio file"""
        result = self.model.transcribe(
            audio_path,
            language=language,
            task="transcribe",
            verbose=False
        )
        return result

class TranslationEngine:
    """Translates text using HuggingFace models"""
    
    def __init__(self):
        self.supported_languages = {
            'es': 'Spanish', 'fr': 'French', 'de': 'German', 
            'it': 'Italian', 'pt': 'Portuguese', 'hi': 'Hindi',
            'zh': 'Chinese', 'ja': 'Japanese', 'ko': 'Korean'
        }
        
        # Use a simpler model that doesn't require sentencepiece initially
        try:
            self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self.use_mbart = True
        except Exception as e:
            print(f"Warning: Could not load mBART model ({str(e)}). Falling back to Helsinki-NLP models.")
            # Fallback to simpler Helsinki models that work without sentencepiece
            self.model_name = "Helsinki-NLP/opus-mt-en-es"  # Default English to Spanish
            self.use_mbart = False
            self.models = {}  # Cache for different language pairs
        
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text from source to target language"""
        try:
            if self.use_mbart:
                # Use mBART model
                self.tokenizer.src_lang = source_lang
                encoded = self.tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
                
                generated_tokens = self.model.generate(
                    **encoded,
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[target_lang],
                    max_length=1024
                )
                
                translation = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            else:
                # Use Helsinki-NLP models (simpler, no sentencepiece needed)
                model_name = f"Helsinki-NLP/opus-mt-en-{target_lang[:2]}"
                
                # Cache models to avoid reloading
                if model_name not in self.models:
                    try:
                        self.models[model_name] = {
                            'tokenizer': AutoTokenizer.from_pretrained(model_name),
                            'model': AutoModelForSeq2SeqLM.from_pretrained(model_name)
                        }
                    except Exception as e:
                        print(f"Could not load {model_name}: {e}")
                        return text  # Return original text if translation fails
                
                tokenizer = self.models[model_name]['tokenizer']
                model = self.models[model_name]['model']
                
                encoded = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
                generated_tokens = model.generate(**encoded, max_length=512)
                translation = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            
            return translation
        except Exception as e:
            print(f"Translation error: {e}")
            return text  # Return original text on error

class VoiceCloner:
    """Clones voice using TTS models"""
    
    def __init__(self):
        # Use XTTS model for voice cloning
        self.tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
        
    def clone_voice(self, text: str, reference_audio: str, output_path: str, language: str = "en") -> str:
        """Clone voice and synthesize speech"""
        try:
            self.tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=reference_audio,
                language=language
            )
            return output_path
        except Exception as e:
            raise Exception(f"Voice cloning error: {str(e)}")

class Summarizer:
    """Generates summaries using transformers"""
    
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.summarizer = pipeline("summarization", model=self.model_name)
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 50) -> str:
        """Generate summary of text"""
        # Split long text into chunks
        max_chunk = 1024
        chunks = [text[i:i+max_chunk] for i in range(0, len(text), max_chunk)]
        
        summaries = []
        for chunk in chunks:
            if len(chunk.split()) < 10:
                continue
            result = self.summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
            summaries.append(result[0]['summary_text'])
        
        return " ".join(summaries)

class QAChatbot:
    """Conversational AI with ChromaDB for context-aware responses"""
    
    def __init__(self, persist_directory: str = "./chromadb"):
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="video_transcripts",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Load QA model
        self.qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    def add_context(self, text: str, metadata: dict = None):
        """Add text to vector database"""
        # Split into sentences
        sentences = text.split('. ')
        
        for idx, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:
                self.collection.add(
                    documents=[sentence],
                    metadatas=[metadata or {}],
                    ids=[f"sent_{idx}_{datetime.now().timestamp()}"]
                )
    
    def query(self, question: str, n_results: int = 5) -> Dict:
        """Query the chatbot with context awareness"""
        # Retrieve relevant context
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            return {"answer": "I don't have enough context to answer that question.", "confidence": 0.0}
        
        # Combine context
        context = " ".join(results['documents'][0])
        
        # Get answer from QA model
        try:
            result = self.qa_model(question=question, context=context)
            return {
                "answer": result['answer'],
                "confidence": result['score'],
                "context": context[:200] + "..."
            }
        except:
            return {"answer": "Unable to find a specific answer in the context.", "confidence": 0.0}

class RedubPipeline:
    """End-to-end pipeline orchestrator"""
    
    def __init__(self):
        self.youtube_processor = YouTubeProcessor()
        self.transcription_engine = TranscriptionEngine()
        self.translation_engine = TranslationEngine()
        self.voice_cloner = VoiceCloner()
        self.summarizer = Summarizer()
        self.chatbot = QAChatbot()
        
        self.stats = {
            'processed_videos': 0,
            'success_rate': 0.0,
            'total_attempts': 0
        }
    
    def process_video(self, url: str, target_language: str = "es", 
                     reference_audio: str = None) -> Dict:
        """Process entire video pipeline"""
        self.stats['total_attempts'] += 1
        
        try:
            # Step 1: Download video
            audio_path, metadata = self.youtube_processor.download_video(url)
            
            # Step 2: Transcribe
            transcription = self.transcription_engine.transcribe(audio_path)
            original_text = transcription['text']
            
            # Step 3: Translate
            translated_text = self.translation_engine.translate(
                original_text, 
                source_lang="en_XX",
                target_lang=f"{target_language}_XX"
            )
            
            # Step 4: Generate summary
            summary = self.summarizer.summarize(original_text)
            
            # Step 5: Voice cloning (if reference provided)
            dubbed_audio_path = None
            if reference_audio:
                output_path = os.path.join(self.youtube_processor.temp_dir, "dubbed_audio.wav")
                dubbed_audio_path = self.voice_cloner.clone_voice(
                    translated_text, 
                    reference_audio, 
                    output_path,
                    language=target_language
                )
            
            # Step 6: Add to chatbot context
            self.chatbot.add_context(original_text, metadata)
            
            # Update stats
            self.stats['processed_videos'] += 1
            self.stats['success_rate'] = (self.stats['processed_videos'] / self.stats['total_attempts']) * 100
            
            return {
                'success': True,
                'metadata': metadata,
                'original_text': original_text,
                'translated_text': translated_text,
                'summary': summary,
                'dubbed_audio': dubbed_audio_path,
                'stats': self.stats
            }
            
        except Exception as e:
            self.stats['success_rate'] = (self.stats['processed_videos'] / self.stats['total_attempts']) * 100
            return {
                'success': False,
                'error': str(e),
                'stats': self.stats
            }

def main():
    st.set_page_config(page_title="YouTube Redubbing System", layout="wide")
    
    st.title("🎥 YouTube Redubbing Multilingual Summarizer with Q&A")
    st.markdown("*Process YouTube videos with transcription, translation, voice cloning, and intelligent Q&A*")
    
    # Initialize pipeline
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = RedubPipeline()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        target_lang = st.selectbox(
            "Target Language",
            ["es", "fr", "de", "it", "pt", "hi", "zh", "ja", "ko"]
        )
        
        reference_audio = st.file_uploader("Upload Reference Voice (Optional)", type=['wav', 'mp3'])
        
        st.markdown("---")
        st.metric("Videos Processed", st.session_state.pipeline.stats['processed_videos'])
        st.metric("Success Rate", f"{st.session_state.pipeline.stats['success_rate']:.1f}%")
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📥 Process Video", "💬 Q&A Chatbot", "📊 Results"])
    
    with tab1:
        st.header("Process YouTube Video")
        youtube_url = st.text_input("Enter YouTube URL:")
        
        if st.button("🚀 Process Video", type="primary"):
            if youtube_url:
                with st.spinner("Processing video... This may take a few minutes."):
                    # Save reference audio if provided
                    ref_audio_path = None
                    if reference_audio:
                        ref_audio_path = os.path.join(tempfile.gettempdir(), "reference.wav")
                        with open(ref_audio_path, 'wb') as f:
                            f.write(reference_audio.getvalue())
                    
                    # Process video
                    result = st.session_state.pipeline.process_video(
                        youtube_url,
                        target_language=target_lang,
                        reference_audio=ref_audio_path
                    )
                    
                    st.session_state.last_result = result
                    
                    if result['success']:
                        st.success("✅ Video processed successfully!")
                    else:
                        st.error(f"❌ Error: {result['error']}")
            else:
                st.warning("Please enter a YouTube URL")
    
    with tab2:
        st.header("Ask Questions About Processed Videos")
        question = st.text_input("Enter your question:")
        
        if st.button("🔍 Get Answer"):
            if question:
                with st.spinner("Searching for answer..."):
                    answer = st.session_state.pipeline.chatbot.query(question)
                    
                    st.markdown("### Answer")
                    st.write(answer['answer'])
                    st.metric("Confidence", f"{answer['confidence']:.2%}")
                    
                    if 'context' in answer:
                        with st.expander("View Context"):
                            st.write(answer['context'])
    
    with tab3:
        if hasattr(st.session_state, 'last_result') and st.session_state.last_result['success']:
            result = st.session_state.last_result
            
            st.header("📋 Processing Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Video Metadata")
                st.json(result['metadata'])
                
                st.subheader("Summary")
                st.write(result['summary'])
            
            with col2:
                st.subheader("Original Transcript")
                with st.expander("View Full Transcript"):
                    st.write(result['original_text'])
                
                st.subheader("Translated Text")
                with st.expander("View Translation"):
                    st.write(result['translated_text'])
            
            if result['dubbed_audio']:
                st.audio(result['dubbed_audio'])
                
                with open(result['dubbed_audio'], 'rb') as f:
                    st.download_button(
                        label="📥 Download Dubbed Audio",
                        data=f,
                        file_name="dubbed_audio.wav",
                        mime="audio/wav"
                    )

if __name__ == "__main__":
    main()
#interface cli
"""
Command Line Interface for YouTube Redubbing System
Provides interactive CLI for video processing
"""

import click
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
from Mainpipline import RedubPipeline
import os
from transformers.generation import BeamSearchScorer

console = Console()

@click.group()
def cli():
    """YouTube Redubbing Multilingual Summarizer with Q&A - CLI Interface"""
    pass

@cli.command()
@click.option('--url', '-u', required=True, help='YouTube video URL')
@click.option('--language', '-l', default='es', help='Target language code (es, fr, de, etc.)')
@click.option('--reference', '-r', help='Path to reference audio file for voice cloning')
@click.option('--output', '-o', default='./output', help='Output directory')
def process(url, language, reference, output):
    """Process a YouTube video"""
    
    console.print(Panel.fit(
        f"[bold cyan]Processing Video[/bold cyan]\n"
        f"URL: {url}\n"
        f"Target Language: {language}\n"
        f"Reference Audio: {reference or 'None'}",
        border_style="cyan"
    ))
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Initialize pipeline
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Initializing pipeline...", total=None)
        pipeline = RedubPipeline()
        
        progress.update(task, description="[cyan]Processing video...")
        result = pipeline.process_video(url, target_language=language, reference_audio=reference)
        
        progress.update(task, description="[green]✓ Complete!", completed=True)
    
    if result['success']:
        console.print("\n[bold green]✓ Processing Successful![/bold green]\n")
        
        # Display metadata
        console.print("[bold]Video Metadata:[/bold]")
        metadata_table = Table(show_header=False)
        for key, value in result['metadata'].items():
            metadata_table.add_row(key.capitalize(), str(value))
        console.print(metadata_table)
        
        # Display summary
        console.print("\n[bold]Summary:[/bold]")
        console.print(Panel(result['summary'], border_style="green"))
        
        # Save outputs
        console.print("\n[bold]Saving outputs...[/bold]")
        
        # Save transcript
        transcript_path = os.path.join(output, 'transcript.txt')
        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(result['original_text'])
        console.print(f"✓ Transcript saved to: {transcript_path}")
        
        # Save translation
        translation_path = os.path.join(output, 'translation.txt')
        with open(translation_path, 'w', encoding='utf-8') as f:
            f.write(result['translated_text'])
        console.print(f"✓ Translation saved to: {translation_path}")
        
        # Save summary
        summary_path = os.path.join(output, 'summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(result['summary'])
        console.print(f"✓ Summary saved to: {summary_path}")
        
        # Save metadata
        metadata_path = os.path.join(output, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result['metadata'], f, indent=2)
        console.print(f"✓ Metadata saved to: {metadata_path}")
        
        if result['dubbed_audio']:
            console.print(f"✓ Dubbed audio available at: {result['dubbed_audio']}")
        
        # Display stats
        console.print(f"\n[bold]Statistics:[/bold]")
        stats_table = Table(show_header=False)
        stats_table.add_row("Success Rate", f"{result['stats']['success_rate']:.1f}%")
        stats_table.add_row("Total Processed", str(result['stats']['processed_videos']))
        console.print(stats_table)
        
    else:
        console.print(f"\n[bold red]✗ Processing Failed![/bold red]")
        console.print(f"Error: {result['error']}")

@cli.command()
@click.option('--question', '-q', required=True, help='Question to ask')
@click.option('--context', '-c', default=5, help='Number of context results')
def qa(question, context):
    """Ask questions about processed videos"""
    
    console.print(Panel.fit(
        f"[bold cyan]Q&A Session[/bold cyan]\n"
        f"Question: {question}",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Searching for answer...", total=None)
        
        pipeline = RedubPipeline()
        answer = pipeline.chatbot.query(question, n_results=context)
        
        progress.update(task, description="[green]✓ Answer found!", completed=True)
    
    console.print("\n[bold]Answer:[/bold]")
    console.print(Panel(answer['answer'], border_style="green"))
    
    console.print(f"\n[bold]Confidence:[/bold] {answer['confidence']:.2%}")
    
    if 'context' in answer:
        console.print("\n[bold]Context Used:[/bold]")
        console.print(answer['context'])

@cli.command()
def languages():
    """List supported languages"""
    
    langs = {
        'es': 'Spanish',
        'fr': 'French',
        'de': 'German',
        'it': 'Italian',
        'pt': 'Portuguese',
        'hi': 'Hindi',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ko': 'Korean'
    }
    
    console.print("\n[bold cyan]Supported Languages:[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="green")
    
    for code, name in langs.items():
        table.add_row(code, name)
    
    console.print(table)

@cli.command()
def stats():
    """Display processing statistics"""
    
    pipeline = RedubPipeline()
    
    console.print("\n[bold cyan]Processing Statistics:[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Videos Processed", str(pipeline.stats['processed_videos']))
    table.add_row("Total Attempts", str(pipeline.stats['total_attempts']))
    table.add_row("Success Rate", f"{pipeline.stats['success_rate']:.1f}%")
    
    console.print(table)

@cli.command()
@click.option('--url', '-u', required=True, help='YouTube video URL')
@click.option('--language', '-l', default='es', help='Target language')
@click.option('--reference', '-r', help='Reference audio path')
def interactive(url, language, reference):
    """Interactive mode with Q&A after processing"""
    
    console.print(Panel.fit(
        "[bold cyan]Interactive Mode[/bold cyan]\n"
        "Process video and then ask questions",
        border_style="cyan"
    ))
    
    # First, process the video
    console.print("\n[bold]Step 1: Processing Video[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        task = progress.add_task("[cyan]Processing...", total=None)
        pipeline = RedubPipeline()
        result = pipeline.process_video(url, target_language=language, reference_audio=reference)
        progress.update(task, description="[green]✓ Complete!", completed=True)
    
    if not result['success']:
        console.print(f"[bold red]Error:[/bold red] {result['error']}")
        return
    
    console.print("\n[bold green]✓ Video processed successfully![/bold green]")
    console.print(Panel(result['summary'], title="Summary", border_style="green"))
    
    # Q&A session
    console.print("\n[bold]Step 2: Q&A Session[/bold]")
    console.print("Ask questions about the video (type 'exit' to quit)")
    
    while True:
        question = click.prompt('\n[?]', type=str)
        
        if question.lower() in ['exit', 'quit', 'q']:
            console.print("[cyan]Goodbye![/cyan]")
            break
        
        answer = pipeline.chatbot.query(question)
        
        console.print(f"\n[bold]Answer:[/bold] {answer['answer']}")
        console.print(f"[dim]Confidence: {answer['confidence']:.2%}[/dim]")

if __name__ == '__main__':
    cli()
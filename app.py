import asyncio
import json
import os
import time
import pyaudio
import sys
import boto3
import sounddevice
import logging

from concurrent.futures import ThreadPoolExecutor
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent, TranscriptResultStream

from api_request_schema import api_request_list, model_ids

model_id = os.getenv('MODEL_ID', 'us.anthropic.claude-3-5-haiku-20241022-v1:0')
aws_region = os.getenv('AWS_REGION', 'us-east-1')

if model_id not in model_ids:
    print(f'Error: Models ID {model_id} in not a valid model ID. Set MODEL_ID env var to one of {model_ids}.')
    sys.exit(0)

api_request = api_request_list[model_id]
config = {
    'log_level': 'none',  # One of: info, debug, none
    'last_speech': "다른 질문이 있으시다면 언제든 말씀해주세요. 좋은 하루 되세요!",
    'region': aws_region,
    'polly': {
        'Engine': 'neural',
        'LanguageCode': 'ko-KR',
        'VoiceId': 'Seoyeon',
        'OutputFormat': 'pcm',
    },
    'bedrock': {
        'response_streaming': True,
        'api_request': api_request
    }
}

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO if config['log_level'] == 'info'
    else logging.DEBUG if config['log_level'] == 'debug'
    else logging.WARNING
)
logger = logging.getLogger(__name__)

p = pyaudio.PyAudio()
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=config['region'])
polly = boto3.client('polly', region_name=config['region'])
transcribe_streaming = TranscribeStreamingClient(region=config['region'])


class UserInputManager:
    """사용자 입력과 실행기 종료 기능을 관리합니다"""
    shutdown_executor = False
    executor = None

    @staticmethod
    def set_executor(executor):
        UserInputManager.executor = executor

    @staticmethod
    def start_shutdown_executor():
        UserInputManager.shutdown_executor = False
        raise Exception()  # Executor shutdown workaround

    @staticmethod
    def start_user_input_loop():
        while True:
            sys.stdin.readline().strip()
            logger.debug('User input to shut down executor...')
            UserInputManager.shutdown_executor = True

    @staticmethod
    def is_executor_set():
        return UserInputManager.executor is not None

    @staticmethod
    def is_shutdown_scheduled():
        return UserInputManager.shutdown_executor


def define_bedrock_body(text):
    """Bedrock API 요청 본문을 생성합니다"""
    model_id = config['bedrock']['api_request']['modelId']
    model_provider = model_id.split('.')[1]
    body = config['bedrock']['api_request']['body']

    if model_provider == 'amazon':
        body['messages'][0]['content'][0]['text'] = text
    elif model_provider == 'anthropic':
        body['messages'][0]['content'] = [
            {
                "type": "text",
                "text": text
            }
        ]
    else:
        raise Exception('Unknown model provider.')
    return body

def get_stream_chunk(event):
    return event.get('chunk')

def get_stream_text(chunk):
    model_id = config['bedrock']['api_request']['modelId']
    model_provider = model_id.split('.')[1]

    chunk_obj = ''
    text = ''
    if model_provider == 'amazon':
        chunk_obj = json.loads(chunk.get('bytes').decode())
        content_block_delta = chunk_obj.get('contentBlockDelta')
        if content_block_delta:
            text = content_block_delta.get('delta').get('text', '')
    elif model_provider == 'anthropic':
        chunk_obj = json.loads(chunk.get('bytes').decode())
        if chunk_obj.get('type') == 'content_block_delta':
            text = chunk_obj['delta'].get('text', '')
    else:
        raise NotImplementedError('Unknown model provider.')

    logger.debug(f'{chunk_obj}')
    return text

def to_audio_generator(bedrock_stream):
    """
    Bedrock의 스트리밍 응답을 오디오 청크로 변환합니다
    문장 경계에서 응답을 분할하여 실시간 스트리밍을 구현합니다
    """
    prefix = ''

    if bedrock_stream:
        for event in bedrock_stream:
            chunk = get_stream_chunk(event)
            if chunk:
                text = get_stream_text(chunk)

                # Split at sentence boundaries for smoother streaming
                if '.' in text:
                    sentences = text.split('.')[:-1]
                    to_polly = ''.join([prefix, '.'.join(sentences), '. '])
                    prefix = text.split('.')[-1]
                    print(to_polly, flush=True, end='')
                    yield to_polly
                else:
                    prefix = ''.join([prefix, text])

        if prefix != '':
            print(prefix, flush=True, end='')
            yield f'{prefix}.'

        print('\n')


class BedrockWrapper:
    """Amazon Bedrock과의 스트리밍 상호작용을 처리합니다"""
    def __init__(self):
        self.speaking = False

    def is_speaking(self):
        return self.speaking

    def invoke_bedrock(self, text):
        """
        Amazon Bedrock LLM 모델을 호출하여 응답을 생성하고 스트리밍합니다
        실시간으로 응답을 오디오로 변환합니다
        """
        logger.debug('Bedrock generation started')
        self.speaking = True

        try:
            body = define_bedrock_body(text)
            body_json = json.dumps(body)
            
            # Bedrock 모델 호출 및 스트리밍 응답 초기화
            response = bedrock_runtime.invoke_model_with_response_stream(
                body=body_json,
                modelId=config['bedrock']['api_request']['modelId'],
                accept=config['bedrock']['api_request']['accept'],
                contentType=config['bedrock']['api_request']['contentType']
            )

            bedrock_stream = response.get('body')
            audio_gen = to_audio_generator(bedrock_stream)

            # Process streaming response in chunks
            reader = Reader()
            for audio in audio_gen:
                reader.read(audio)

            reader.close()

        except Exception as e:
            print(e)
            time.sleep(2)
            self.speaking = False

        time.sleep(1)
        self.speaking = False
        logger.debug('Bedrock generation completed')


class Reader:
    """PyAudio를 사용하여 오디오 스트리밍 출력을 처리합니다"""
    def __init__(self):
        self.audio = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        self.chunk = 1024

    def read(self, data):
        """
        실시간 재생을 위해 청크 단위로 오디오 데이터를 스트리밍합니다
        원활한 스트리밍을 위한 중단 처리를 구현합니다
        """
        response = polly.synthesize_speech(
            Text=data,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )

        stream = response['AudioStream']

        while True:
            if UserInputManager.is_executor_set() and UserInputManager.is_shutdown_scheduled():
                UserInputManager.start_shutdown_executor()

            data = stream.read(self.chunk)
            self.audio.write(data)
            if not data:
                break

    def close(self):
        time.sleep(1)
        self.audio.stop_stream()
        self.audio.close()


def aws_polly_tts(polly_text):
    """
    Amazon Polly를 사용하여 텍스트를 음성으로 변환합니다
    실시간 응답을 위해 문장 단위로 처리합니다
    """
    logger.debug(f'Character count: {len(polly_text)}')
    chunk_size = 1024
    sentences = polly_text.split('. ')
    
    # 실시간 응답을 위해 20문장씩 처리
    for i in range(0, len(sentences), 20):
        current_chunk = '. '.join(sentences[i:i + 20])
        logger.debug(f'Processing sentences {i} to {i + 20}')
        
        if not current_chunk.strip():
            continue
            
        # Polly TTS 호출
        response = polly.synthesize_speech(
            Text=current_chunk,
            Engine=config['polly']['Engine'],
            LanguageCode=config['polly']['LanguageCode'],
            VoiceId=config['polly']['VoiceId'],
            OutputFormat=config['polly']['OutputFormat'],
        )
        
        audio_stream = response['AudioStream']
        polly_stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, output=True)
        
        # Stream in chunks
        while True:
            data = audio_stream.read(chunk_size)
            if not data:
                break
            polly_stream.write(data)
        
        audio_stream.close()
        polly_stream.stop_stream()
        polly_stream.close()
        time.sleep(0.1)  # Add small delay between sentences


class EventHandler(TranscriptResultStreamHandler):
    def __init__(self, transcript_result_stream: TranscriptResultStream, bedrock_wrapper):
        super().__init__(transcript_result_stream)
        self.bedrock_wrapper = bedrock_wrapper
        self.text = []
        self.sample_count = 0
        self.max_sample_counter = 120  # Wait for 2 seconds
        self.silence_threshold = 10    # Quick processing after speech ends

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        results = transcript_event.transcript.results
        if not self.bedrock_wrapper.is_speaking():
            if results:
                for result in results:
                    self.sample_count = 0
                    if not result.is_partial:
                        for alt in result.alternatives:
                            print(alt.transcript, flush=True, end=' ')
                            self.text.append(alt.transcript)
            else:
                self.sample_count += 1
                if (len(self.text) > 0 and self.sample_count >= self.silence_threshold) or \
                   (len(self.text) == 0 and self.sample_count >= self.max_sample_counter):
                    if len(self.text) == 0:
                        last_speech = config['last_speech']
                        print(last_speech, flush=True)
                        aws_polly_tts(last_speech)
                        os._exit(0)
                    else:
                        input_text = ' '.join(self.text)
                        logger.info(f'\nUser input: {input_text}')

                        executor = ThreadPoolExecutor(max_workers=1)
                        UserInputManager.set_executor(executor)
                        loop.run_in_executor(
                            executor,
                            self.bedrock_wrapper.invoke_bedrock,
                            input_text
                        )

                    self.text.clear()
                    self.sample_count = 0


class MicStream:
    async def mic_stream(self):
        loop = asyncio.get_event_loop()
        input_queue = asyncio.Queue()

        def callback(indata, frame_count, time_info, status):
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))

        stream = sounddevice.RawInputStream(
            channels=1, samplerate=16000, callback=callback, blocksize=2048 * 2, dtype="int16")
        with stream:
            while True:
                indata, status = await input_queue.get()
                yield indata, status

    async def write_chunks(self, stream):
        async for chunk, status in self.mic_stream():
            await stream.input_stream.send_audio_event(audio_chunk=chunk)

        await stream.input_stream.end_stream()

    async def basic_transcribe(self):
        """
        Amazon Transcribe를 사용하여 실시간 음성-텍스트 변환을 수행합니다
        """
        loop.run_in_executor(ThreadPoolExecutor(max_workers=1), UserInputManager.start_user_input_loop)

        stream = await transcribe_streaming.start_stream_transcription(
            language_code="ko-KR",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )

        handler = EventHandler(stream.output_stream, BedrockWrapper())
        await asyncio.gather(self.write_chunks(stream), handler.handle_events())


info_text = f'''
*************************************************************
[정보] 지원되는 FM 모델: {model_ids}.
[정보] FM 모델 변경은 <MODEL_ID> 환경변수로 설정하세요. 예시: export MODEL_ID=meta.llama2-70b-chat-v1

[정보] AWS 리전: {config['region']}
[정보] Amazon Bedrock 모델: {config['bedrock']['api_request']['modelId']}
[정보] Polly 설정: 엔진 {config['polly']['Engine']}, 음성 {config['polly']['VoiceId']}
[정보] 로그 레벨: {config['log_level']}

[정보] Amazon Bedrock을 중단하려면 Enter 키를 누르세요. 그 후 계속 대화할 수 있습니다!
[정보] Amazon Bedrock과 음성 대화를 시작하세요!
*************************************************************
'''
print(info_text)

loop = asyncio.get_event_loop()
try:
    loop.run_until_complete(MicStream().basic_transcribe())
except (KeyboardInterrupt, Exception) as e:
    print()

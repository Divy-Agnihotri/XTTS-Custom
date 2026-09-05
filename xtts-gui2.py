import sys
import json
import re

import torch
import torchaudio


from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QDial,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QScrollArea,
    QTabWidget,
)

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


SAMPLE_RATE = 24000


class XTTSGui(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Sangya")
        self.resize(900, 800)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_dir = r"D:\TTS\models\models--coqui--XTTS-v2\snapshots\6c2b0d75eae4b7047358e3b6bd9325f857d43f77"

        self.audio_path = ""

        # Play-mode state
        self.play_data = None
        self.character_voices = {}          # character name -> wav path
        self.character_voice_rows = {}      # character name -> label widget

        self.init_ui()

        self.log("Loading XTTS model...")
        self.load_model()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def init_ui(self):

        root_layout = QVBoxLayout()

        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        self.tabs.addTab(self.build_single_tab(), "Single Line")
        self.tabs.addTab(self.build_play_tab(), "Play Mode")

        # Shared knobs (used by both tabs)
        knob_layout = QHBoxLayout()

        self.temp_dial = self.create_dial("Temperature", 0.1, 1.0, 0.75, float)
        self.speed_dial = self.create_dial("Speed", 0.5, 2.0, 1.0, float)
        self.topk_dial = self.create_dial("Top K", 1, 200, 50, int)
        self.topp_dial = self.create_dial("Top P", 0.5, 1.0, 0.85, float)
        self.rep_dial = self.create_dial("Repetition Penalty", 1.0, 15.0, 10.0, float)
        self.pause_dial = self.create_dial("Line Pause (s)", 0.0, 3.0, 0.4, float)

        for d in (
            self.temp_dial,
            self.speed_dial,
            self.topk_dial,
            self.topp_dial,
            self.rep_dial,
            self.pause_dial,
        ):
            knob_layout.addWidget(d["widget"])

        root_layout.addLayout(knob_layout)

        # Output filename (shared)
        root_layout.addWidget(QLabel("Output Filename"))
        self.output_file = QLineEdit()
        self.output_file.setText("output.wav")
        root_layout.addWidget(self.output_file)

        # Status log (shared)
        root_layout.addWidget(QLabel("Status"))
        self.status = QTextEdit()
        self.status.setReadOnly(True)
        root_layout.addWidget(self.status)

        self.setLayout(root_layout)

    def build_single_tab(self):

        tab = QWidget()
        layout = QVBoxLayout()

        audio_layout = QHBoxLayout()
        self.audio_label = QLabel("No audio selected")
        btn_audio = QPushButton("Select Reference Audio")
        btn_audio.clicked.connect(self.select_audio)
        audio_layout.addWidget(btn_audio)
        audio_layout.addWidget(self.audio_label)
        layout.addLayout(audio_layout)

        self.language = QComboBox()
        self.language.addItems(
            ["hi", "en", "fr", "de", "es", "it", "ja", "ko", "pt", "ru", "zh-cn"]
        )
        layout.addWidget(QLabel("Language"))
        layout.addWidget(self.language)

        layout.addWidget(QLabel("Text"))
        self.text_input = QTextEdit()
        layout.addWidget(self.text_input)

        btn_generate = QPushButton("Generate Speech")
        btn_generate.clicked.connect(self.generate)
        layout.addWidget(btn_generate)

        tab.setLayout(layout)
        return tab

    def build_play_tab(self):

        tab = QWidget()
        layout = QVBoxLayout()

        # Load play JSON
        load_layout = QHBoxLayout()
        btn_load_play = QPushButton("Load Play JSON")
        btn_load_play.clicked.connect(self.load_play)
        self.play_info_label = QLabel("No play loaded")
        load_layout.addWidget(btn_load_play)
        load_layout.addWidget(self.play_info_label)
        layout.addLayout(load_layout)

        self.play_language = QComboBox()
        self.play_language.addItems(
            ["hi", "en", "fr", "de", "es", "it", "ja", "ko", "pt", "ru", "zh-cn"]
        )
        layout.addWidget(QLabel("Language"))
        layout.addWidget(self.play_language)

        # Scrollable character -> voice mapping area
        layout.addWidget(QLabel("Character Voices"))

        self.char_voice_container = QWidget()
        self.char_voice_layout = QVBoxLayout()
        self.char_voice_container.setLayout(self.char_voice_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.char_voice_container)
        scroll.setMinimumHeight(200)
        layout.addWidget(scroll)

        btn_generate_play = QPushButton("Generate Play")
        btn_generate_play.clicked.connect(self.generate_play)
        layout.addWidget(btn_generate_play)

        tab.setLayout(layout)
        return tab

    def create_dial(self, title, min_val, max_val, default, value_type=float, resolution=100):

        container = QWidget()
        v = QVBoxLayout(container)

        lbl_name = QLabel(title)
        dial = QDial()
        dial.setMinimum(0)
        dial.setMaximum(resolution)

        if value_type == float:
            default_pos = int(((default - min_val) / (max_val - min_val)) * resolution)
        else:
            default_pos = default - min_val
            dial.setMaximum(max_val - min_val)

        dial.setValue(default_pos)

        lbl_value = QLabel(str(default))

        def get_actual_value():
            if value_type == float:
                value = min_val + ((dial.value() / resolution) * (max_val - min_val))
                return round(value, 3)
            return int(min_val + dial.value())

        def update_label():
            lbl_value.setText(str(get_actual_value()))

        dial.valueChanged.connect(lambda _: update_label())
        update_label()

        v.addWidget(lbl_name)
        v.addWidget(dial)
        v.addWidget(lbl_value)

        return {
            "widget": container,
            "dial": dial,
            "label": lbl_value,
            "get_value": get_actual_value,
        }

    def log(self, text):
        self.status.append(text)
        QApplication.processEvents()

    # ------------------------------------------------------------------
    # Model / audio selection
    # ------------------------------------------------------------------

    def select_audio(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Audio", "", "Audio Files (*.wav *.mp3)"
        )
        if file_name:
            self.audio_path = file_name
            self.audio_label.setText(file_name)

    def load_model(self):
        try:
            config = XttsConfig()
            config.load_json(f"{self.model_dir}\\config.json")

            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(config, checkpoint_dir=self.model_dir, eval=True)
            self.model.to(self.device)

            self.log(f"Model loaded successfully ({self.device})")

        except Exception as e:
            self.log(str(e))

    # ------------------------------------------------------------------
    # Text chunking (shared by single-line and play generation)
    # ------------------------------------------------------------------

    def split_into_chunks(self, text, max_len=150):

        sentences = re.findall(r'[^।!?.]+[।!?.]?', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:

            sentence = sentence.strip()
            if not sentence:
                continue

            if len(current_chunk) + len(sentence) + 1 <= max_len:
                current_chunk = sentence if not current_chunk else current_chunk + " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                if len(sentence) <= max_len:
                    current_chunk = sentence
                else:
                    start = 0
                    while start < len(sentence):
                        part = sentence[start:start + max_len]

                        if len(part) < max_len:
                            chunks.append(part.strip())
                            break

                        last_space = part.rfind(" ")

                        if last_space > 0:
                            chunks.append(part[:last_space].strip())
                            start += last_space + 1
                        else:
                            chunks.append(part)
                            start += max_len

        if current_chunk:
            chunks.append(current_chunk)

        return [c for c in chunks if c.strip()]

    # ------------------------------------------------------------------
    # Single-line generation (original behaviour)
    # ------------------------------------------------------------------

    def generate(self):

        try:
            if not self.audio_path:
                QMessageBox.warning(self, "Error", "Select a reference audio first.")
                return

            text = self.text_input.toPlainText().strip()

            if not text:
                QMessageBox.warning(self, "Error", "Enter text.")
                return

            chunks = self.split_into_chunks(text)

            self.log("Creating speaker embedding...")

            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                audio_path=[self.audio_path]
            )

            temperature = self.temp_dial["get_value"]()
            speed = self.speed_dial["get_value"]()
            top_k = self.topk_dial["get_value"]()
            top_p = self.topp_dial["get_value"]()
            repetition_penalty = self.rep_dial["get_value"]()

            self.log("Running inference...")

            all_audio = []

            for chunk in chunks:
                self.log(f"Generating chunk: {chunk[:50]}...")

                out = self.model.inference(
                    chunk,
                    self.language.currentText(),
                    gpt_cond_latent,
                    speaker_embedding,
                    temperature=temperature,
                    speed=speed,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    enable_text_splitting=False,
                )

                all_audio.extend(out["wav"])

            self._save_audio(all_audio, self.output_file.text())

        except Exception as e:
            self.log(str(e))
            QMessageBox.critical(self, "Error", str(e))

    # ------------------------------------------------------------------
    # Play mode
    # ------------------------------------------------------------------

    def load_play(self):

        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Play JSON", "", "JSON Files (*.json)"
        )

        if not file_name:
            return

        try:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)

            characters = data.get("characters", [])
            play = data.get("play", [])

            if not characters or not play:
                QMessageBox.warning(
                    self, "Error", "JSON must contain 'characters' and 'play' lists."
                )
                return

            self.play_data = data
            self.build_character_voice_rows(characters)

            self.play_info_label.setText(
                f"Loaded: {len(play)} lines, {len(characters)} characters"
            )
            self.log(
                f"Loaded play '{file_name}' — {len(play)} lines, "
                f"{len(characters)} characters"
            )

        except Exception as e:
            self.log(str(e))
            QMessageBox.critical(self, "Error", str(e))

    def build_character_voice_rows(self, characters):

        # clear previous rows
        while self.char_voice_layout.count():
            item = self.char_voice_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.character_voices = {}
        self.character_voice_rows = {}

        for char in characters:

            row_widget = QWidget()
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)

            name_label = QLabel(char)
            name_label.setMinimumWidth(120)

            path_label = QLabel("No voice selected")

            btn = QPushButton("Select Voice")
            # default-arg trick so each lambda captures its own char/path_label
            btn.clicked.connect(
                lambda _, c=char, pl=path_label: self.select_character_voice(c, pl)
            )

            row.addWidget(name_label)
            row.addWidget(btn)
            row.addWidget(path_label)
            row_widget.setLayout(row)

            self.char_voice_layout.addWidget(row_widget)

            self.character_voices[char] = ""
            self.character_voice_rows[char] = path_label

    def select_character_voice(self, character, label_widget):

        file_name, _ = QFileDialog.getOpenFileName(
            self, f"Select Voice for {character}", "", "Audio Files (*.wav *.mp3)"
        )

        if file_name:
            self.character_voices[character] = file_name
            label_widget.setText(file_name)

    def generate_play(self):

        try:
            if not self.play_data:
                QMessageBox.warning(self, "Error", "Load a play JSON first.")
                return

            characters = self.play_data.get("characters", [])
            lines = self.play_data.get("play", [])

            missing = [c for c in characters if not self.character_voices.get(c)]
            if missing:
                QMessageBox.warning(
                    self,
                    "Error",
                    "Missing voice sample(s) for: " + ", ".join(missing),
                )
                return

            temperature = self.temp_dial["get_value"]()
            speed = self.speed_dial["get_value"]()
            top_k = self.topk_dial["get_value"]()
            top_p = self.topp_dial["get_value"]()
            repetition_penalty = self.rep_dial["get_value"]()
            pause_seconds = self.pause_dial["get_value"]()
            language = self.play_language.currentText()

            # Compute each character's conditioning latents once and reuse
            self.log("Preparing voice embeddings for each character...")

            latents_cache = {}
            for char in characters:
                self.log(f"  Embedding voice: {char}")
                gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
                    audio_path=[self.character_voices[char]]
                )
                latents_cache[char] = (gpt_cond_latent, speaker_embedding)

            pause_samples = int(pause_seconds * SAMPLE_RATE)
            silence = [0.0] * pause_samples

            all_audio = []

            self.log(f"Generating {len(lines)} lines...")

            for i, line in enumerate(lines):

                speaker = line.get("speaker")
                dialogue = (line.get("dialogue") or "").strip()

                if speaker not in latents_cache:
                    self.log(
                        f"[{i + 1}/{len(lines)}] Skipping — unknown speaker '{speaker}'"
                    )
                    continue

                if not dialogue:
                    continue

                self.log(f"[{i + 1}/{len(lines)}] {speaker}: {dialogue[:60]}")

                chunks = self.split_into_chunks(dialogue)
                gpt_cond_latent, speaker_embedding = latents_cache[speaker]

                for chunk in chunks:
                    out = self.model.inference(
                        chunk,
                        language,
                        gpt_cond_latent,
                        speaker_embedding,
                        temperature=temperature,
                        speed=speed,
                        top_k=top_k,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        enable_text_splitting=False,
                    )

                    all_audio.extend(out["wav"])

                if pause_samples:
                    all_audio.extend(silence)

            self._save_audio(all_audio, self.output_file.text())

        except Exception as e:
            self.log(str(e))
            QMessageBox.critical(self, "Error", str(e))

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _save_audio(self, samples, output_file):

        if not samples:
            QMessageBox.warning(self, "Error", "No audio was generated.")
            return

        torchaudio.save(
            output_file,
            torch.tensor(samples).unsqueeze(0),
            SAMPLE_RATE,
        )

        self.log(f"Completed. Saved: {output_file}")

        QMessageBox.information(self, "Success", f"Saved to:\n{output_file}")


if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = XTTSGui()
    window.show()

    sys.exit(app.exec())
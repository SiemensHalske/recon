"""
File: recon/container.py
Author: Hendrik Siemens
Date: 2025-05-11
Modified: 2025-05-11

Description:
    This module provides a class to record audio from a WebSDR server.
    It uses the KiwiSDR client to connect to the server and record audio streams.
    
Python Version: >3.8
Dependencies: rich, kiwiclient repo

License: MIT License
"""

import os
import subprocess
import time
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import track


class WebSDR:
    """
    A class to record audio from a WebSDR server using the KiwiSDR client.
    """

    def __init__(self, server, port, freq, mode, basename, tlimit):
        self.server = server
        self.port = port
        self.freq = freq

        # inspect 'self.freq' if it contains more than 1 frequency
        if ',' in self.freq:
            self.freq = self.freq.split(',')
        elif '-' in self.freq:
            start_freq, end_freq = self.freq.split('-')
            self.freq = [
                str(f)
                for f in range(
                    int(start_freq),
                    int(end_freq) + 1
                )
            ]
        else:
            self.freq = [self.freq]
        self.freq = [f.strip() for f in self.freq]

        self.mode = mode
        self.basename = basename
        self.tlimit = tlimit
        self.console = Console()
        self.output_dir = self._create_output_dir()
        self.wav_filename = os.path.join(self.output_dir, "audio.wav")

    def _create_output_dir(self) -> str:
        """Create a timestamped output directory for the recordings."""
        if not os.path.exists("results"):
            os.makedirs("results")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for freq in self.freq:
            dir_name = f"{self.basename}_{freq}_{timestamp}"
            output_dir = os.path.join("results", dir_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                return output_dir
        raise FileExistsError("Failed to create a unique output directory.")

    def get_output_path(self, freq):
        """Generate a unique output path for the recording."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name = f"{self.basename}_{freq}_{timestamp}"
        output_dir = os.path.join("results", dir_name)
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    def record_audio(self):
        """Record audio from the WebSDR server for each frequency."""
        for freq in self.freq:
            output_dir = self.get_output_path(freq)
            wav_filename = os.path.join(output_dir, "audio.wav")
            self.console.rule(
                f"[bold green] Recon: Audio Recording ({freq} kHz)")
            self.console.print(Panel.fit(
                f"[1/1] Recording AUDIO WAV...\n"
                f"Frequency: [bold cyan]{freq}[/bold cyan] kHz\n"
                f"Duration: [bold cyan]{self.tlimit}[/bold cyan] sec\n"
                f"Output: [bold yellow]{wav_filename}[/bold yellow]"
            ))

            process = subprocess.Popen([
                "python", "kiwiclient/kiwirecorder.py",
                "-s", self.server,
                "-p", self.port,
                "-f", freq,
                "-m", self.mode,
                "--fn", wav_filename,
                "--tlimit", self.tlimit,
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            start_time = time.time()
            total_time = int(self.tlimit)

            for line in track(range(total_time), description=f"Recording {freq} kHz..."):
                elapsed_time = time.time() - start_time
                remaining_time = max(0, total_time - int(elapsed_time))

                self.console.print(
                    f"[bold cyan]Step:[/bold cyan] {line + 1}/{total_time} | "
                    f"[bold cyan]Elapsed:[/bold cyan] {int(elapsed_time)} sec | "
                    f"[bold cyan]Remaining:[/bold cyan] {remaining_time} sec"
                )

                # Read process output stream
                if process.stdout:
                    output = process.stdout.readline()
                    if output:
                        self.console.print(
                            f"[bold green]{output.strip()}[/bold green]")

                # Read process error stream
                if process.stderr:
                    error = process.stderr.readline()
                    if error:
                        self.console.print(
                            f"[bold red]{error.strip()}[/bold red]")

                time.sleep(1)  # Simulate progress based on subprocess output

            process.wait()  # Ensure the subprocess finishes

            self.console.rule(f"[bold green] Recon Complete ({freq} kHz)")
            self.console.print(
                Panel.fit(
                    f"[bold green]Recording done![/bold green]\n"
                    f"Saved audio to: [bold yellow]{wav_filename}[/bold yellow]"
                )
            )


if __name__ == "__main__":
    recorder = WebSDR(
        server="g4wim.proxy.kiwisdr.com",
        port="8073",
        freq="25382.58",
        mode="usb",
        basename="test25382-58",
        tlimit="240"
    )
    recorder.record_audio()

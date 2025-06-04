# music

This repository contains a simple ear training tool that can analyze melodies
from `.wav` files and search for similar phrases using solfège syllables.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place your `.wav` songs inside the `songs/` directory.

## Usage

Run the script by specifying the directory with songs and the query file:

```bash
python ear_training.py songs path/to/query.wav
```

The tool will print songs with melodies similar to the query.

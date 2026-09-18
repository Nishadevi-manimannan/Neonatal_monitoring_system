import os
import csv
import math
import struct
from pathlib import Path


# ============================================================
# NEONATAL CRY DATASET PREPROCESSING
# ============================================================

DATASET_FOLDER = Path(".")

OUTPUT_FOLDER = Path("Preprocessing_Output")

CLASSES = [
    "Hunger",
    "Discomfort",
    "Pain",
    "Sleep"
]

TARGET_SAMPLE_RATE = 16000


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_FOLDER.mkdir(exist_ok=True)


# ============================================================
# READ WAV FILE
# ============================================================

def read_wav(filename):

    with open(filename, "rb") as f:
        data = f.read()

    if len(data) < 12:
        raise ValueError("File is too small")

    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("Not a valid WAV file")

    pos = 12

    audio_format = None
    channels = None
    sample_rate = None
    bits_per_sample = None
    audio_data = None

    while pos + 8 <= len(data):

        chunk_id = data[pos:pos + 4]

        chunk_size = struct.unpack_from(
            "<I",
            data,
            pos + 4
        )[0]

        chunk_start = pos + 8
        chunk_end = chunk_start + chunk_size

        if chunk_end > len(data):
            chunk_end = len(data)

        # ----------------------------------------------------
        # FORMAT CHUNK
        # ----------------------------------------------------

        if chunk_id == b"fmt ":

            if chunk_size < 16:
                raise ValueError("Invalid fmt chunk")

            audio_format = struct.unpack_from(
                "<H",
                data,
                chunk_start
            )[0]

            channels = struct.unpack_from(
                "<H",
                data,
                chunk_start + 2
            )[0]

            sample_rate = struct.unpack_from(
                "<I",
                data,
                chunk_start + 4
            )[0]

            bits_per_sample = struct.unpack_from(
                "<H",
                data,
                chunk_start + 14
            )[0]

            # ------------------------------------------------
            # WAVE_FORMAT_EXTENSIBLE
            # ------------------------------------------------

            if audio_format == 65534 and chunk_size >= 40:

                subformat = data[
                    chunk_start + 24:
                    chunk_start + 40
                ]

                pcm_guid = bytes.fromhex(
                    "0100000000001000800000aa00389b71"
                )

                float_guid = bytes.fromhex(
                    "0300000000001000800000aa00389b71"
                )

                if subformat == pcm_guid:
                    audio_format = 1

                elif subformat == float_guid:
                    audio_format = 3

        # ----------------------------------------------------
        # AUDIO DATA CHUNK
        # ----------------------------------------------------

        elif chunk_id == b"data":

            audio_data = data[
                chunk_start:chunk_end
            ]

        pos = chunk_start + chunk_size

        # WAV chunks are aligned to even boundaries
        if pos % 2 == 1:
            pos += 1

    if audio_data is None:
        raise ValueError("Audio data not found")

    if audio_format not in [1, 3]:
        raise ValueError(
            "Unsupported WAV format: "
            + str(audio_format)
        )

    if channels is None:
        raise ValueError("Channel information missing")

    if sample_rate is None:
        raise ValueError("Sample rate missing")

    # ========================================================
    # PCM INTEGER
    # ========================================================

    samples = []

    if audio_format == 1:

        # ----------------------------------------------------
        # 8 BIT
        # ----------------------------------------------------

        if bits_per_sample == 8:

            for value in audio_data:

                samples.append(
                    (value - 128) / 128.0
                )

        # ----------------------------------------------------
        # 16 BIT
        # ----------------------------------------------------

        elif bits_per_sample == 16:

            count = len(audio_data) // 2

            values = struct.unpack(
                "<" + ("h" * count),
                audio_data[:count * 2]
            )

            for value in values:

                samples.append(
                    value / 32768.0
                )

        # ----------------------------------------------------
        # 24 BIT
        # ----------------------------------------------------

        elif bits_per_sample == 24:

            count = len(audio_data) // 3

            for i in range(count):

                b1 = audio_data[i * 3]
                b2 = audio_data[i * 3 + 1]
                b3 = audio_data[i * 3 + 2]

                value = (
                    b1
                    | (b2 << 8)
                    | (b3 << 16)
                )

                if value & 0x800000:
                    value -= 0x1000000

                samples.append(
                    value / 8388608.0
                )

        # ----------------------------------------------------
        # 32 BIT
        # ----------------------------------------------------

        elif bits_per_sample == 32:

            count = len(audio_data) // 4

            values = struct.unpack(
                "<" + ("i" * count),
                audio_data[:count * 4]
            )

            for value in values:

                samples.append(
                    value / 2147483648.0
                )

        else:

            raise ValueError(
                "Unsupported PCM bit depth: "
                + str(bits_per_sample)
            )

    # ========================================================
    # IEEE FLOAT
    # ========================================================

    elif audio_format == 3:

        if bits_per_sample == 32:

            count = len(audio_data) // 4

            values = struct.unpack(
                "<" + ("f" * count),
                audio_data[:count * 4]
            )

            for value in values:

                if math.isnan(value) or math.isinf(value):
                    value = 0.0

                samples.append(value)

        elif bits_per_sample == 64:

            count = len(audio_data) // 8

            values = struct.unpack(
                "<" + ("d" * count),
                audio_data[:count * 8]
            )

            for value in values:

                if math.isnan(value) or math.isinf(value):
                    value = 0.0

                samples.append(value)

        else:

            raise ValueError(
                "Unsupported floating-point format"
            )

    # ========================================================
    # MULTI CHANNEL → MONO
    # ========================================================

    if channels > 1:

        frame_count = len(samples) // channels

        mono = []

        for i in range(frame_count):

            total = 0.0

            for channel in range(channels):

                total += samples[
                    i * channels + channel
                ]

            mono.append(
                total / channels
            )

        samples = mono

    return (
        samples,
        sample_rate,
        channels,
        bits_per_sample
    )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(samples):

    if not samples:
        return []

    maximum = 0.0

    for value in samples:

        if abs(value) > maximum:
            maximum = abs(value)

    if maximum == 0:
        return samples[:]

    normalized = []

    for value in samples:

        normalized.append(
            value / maximum
        )

    return normalized


# ============================================================
# RESAMPLING
# ============================================================

def resample(
    samples,
    old_rate,
    new_rate
):

    if not samples:
        return []

    if old_rate == new_rate:
        return samples[:]

    new_length = int(
        len(samples)
        * new_rate
        / old_rate
    )

    if new_length < 1:
        new_length = 1

    result = []

    for i in range(new_length):

        if new_length == 1:

            position = 0

        else:

            position = (
                i
                * (len(samples) - 1)
                / (new_length - 1)
            )

        left = int(position)

        right = min(
            left + 1,
            len(samples) - 1
        )

        fraction = position - left

        value = (
            samples[left] * (1 - fraction)
            +
            samples[right] * fraction
        )

        result.append(value)

    return result


# ============================================================
# RMS
# ============================================================

def calculate_rms(samples):

    if not samples:
        return 0.0

    total = 0.0

    for value in samples:

        total += value * value

    return math.sqrt(
        total / len(samples)
    )


# ============================================================
# ZERO CROSSING RATE
# ============================================================

def calculate_zcr(samples):

    if len(samples) < 2:
        return 0.0

    crossings = 0

    for i in range(1, len(samples)):

        previous = samples[i - 1]
        current = samples[i]

        if (
            previous >= 0 and current < 0
        ) or (
            previous < 0 and current >= 0
        ):

            crossings += 1

    return crossings / (
        len(samples) - 1
    )


# ============================================================
# SELECT POINTS FOR GRAPH
#
# IMPORTANT:
# These are REAL samples from the audio.
# They are NOT randomly generated.
# ============================================================

def select_graph_points(samples, number=1500):

    if not samples:
        return []

    if len(samples) <= number:
        return samples[:]

    result = []

    step = len(samples) / number

    for i in range(number):

        index = int(i * step)

        if index >= len(samples):
            index = len(samples) - 1

        result.append(
            samples[index]
        )

    return result


# ============================================================
# CREATE SVG GRAPH
# ============================================================

def create_svg(
    original,
    processed,
    class_name,
    filename,
    original_rate
):

    width = 1400
    height = 850

    left = 110
    right = 1330

    graph_width = right - left

    # --------------------------------------------------------
    # Use actual samples from the recording
    # --------------------------------------------------------

    original_points_data = select_graph_points(
        original
    )

    processed_points_data = select_graph_points(
        processed
    )

    # --------------------------------------------------------
    # Convert samples into SVG coordinates
    # --------------------------------------------------------

    def waveform_points(
        samples,
        top,
        bottom
    ):

        if not samples:
            return ""

        center = (
            top + bottom
        ) / 2

        amplitude = (
            bottom - top
        ) * 0.45

        points = []

        for i, value in enumerate(samples):

            x = (
                left
                +
                (
                    i
                    /
                    max(
                        1,
                        len(samples) - 1
                    )
                )
                * graph_width
            )

            # Clamp amplitude for display
            if value > 1:
                value = 1

            if value < -1:
                value = -1

            y = (
                center
                -
                value * amplitude
            )

            points.append(
                f"{x:.2f},{y:.2f}"
            )

        return " ".join(points)

    original_points = waveform_points(
        original_points_data,
        160,
        390
    )

    processed_points = waveform_points(
        processed_points_data,
        500,
        730
    )

    # --------------------------------------------------------
    # Duration
    # --------------------------------------------------------

    original_duration = (
        len(original)
        / original_rate
    )

    processed_duration = (
        len(processed)
        / TARGET_SAMPLE_RATE
    )

    # --------------------------------------------------------
    # SVG
    # --------------------------------------------------------

    svg = f"""<svg
xmlns="http://www.w3.org/2000/svg"
width="{width}"
height="{height}"
viewBox="0 0 {width} {height}">

<!-- Background -->

<rect
x="0"
y="0"
width="{width}"
height="{height}"
fill="white"/>


<!-- Main title -->

<text
x="700"
y="45"
font-family="Arial"
font-size="28"
font-weight="bold"
text-anchor="middle">

{class_name} Cry - Original vs Preprocessed

</text>


<!-- Dataset information -->

<text
x="110"
y="85"
font-family="Arial"
font-size="18">

Actual dataset file: {filename}

</text>


<text
x="110"
y="112"
font-family="Arial"
font-size="16">

Original sample rate: {original_rate} Hz

</text>


<text
x="500"
y="112"
font-family="Arial"
font-size="16">

Processed sample rate: {TARGET_SAMPLE_RATE} Hz

</text>


<!-- ORIGINAL -->

<text
x="110"
y="145"
font-family="Arial"
font-size="21"
font-weight="bold">

Original Audio Waveform

</text>


<!-- Original axes -->

<line
x1="{left}"
y1="275"
x2="{right}"
y2="275"
stroke="#999999"
stroke-width="1"/>

<line
x1="{left}"
y1="160"
x2="{left}"
y2="390"
stroke="#999999"
stroke-width="1"/>


<!-- Original waveform -->

<polyline
points="{original_points}"
fill="none"
stroke="#222222"
stroke-width="1.3"/>


<!-- Original labels -->

<text
x="45"
y="280"
font-family="Arial"
font-size="14">

0

</text>


<text
x="35"
y="170"
font-family="Arial"
font-size="14">

+1

</text>


<text
x="35"
y="390"
font-family="Arial"
font-size="14">

-1

</text>


<text
x="700"
y="420"
font-family="Arial"
font-size="15"
text-anchor="middle">

Time → ({original_duration:.2f} seconds)

</text>


<!-- PREPROCESSED -->

<text
x="110"
y="485"
font-family="Arial"
font-size="21"
font-weight="bold">

Preprocessed Audio Waveform

</text>


<!-- Processed axes -->

<line
x1="{left}"
y1="615"
x2="{right}"
y2="615"
stroke="#999999"
stroke-width="1"/>

<line
x1="{left}"
y1="500"
x2="{left}"
y2="730"
stroke="#999999"
stroke-width="1"/>


<!-- Processed waveform -->

<polyline
points="{processed_points}"
fill="none"
stroke="#222222"
stroke-width="1.3"/>


<!-- Processed labels -->

<text
x="45"
y="620"
font-family="Arial"
font-size="14">

0

</text>


<text
x="35"
y="510"
font-family="Arial"
font-size="14">

+1

</text>


<text
x="35"
y="730"
font-family="Arial"
font-size="14">

-1

</text>


<text
x="700"
y="760"
font-family="Arial"
font-size="15"
text-anchor="middle">

Time → ({processed_duration:.2f} seconds)

</text>


<!-- PROCESSING INFORMATION -->

<text
x="700"
y="805"
font-family="Arial"
font-size="16"
text-anchor="middle">

Preprocessing: Mono Conversion → 16 kHz Resampling → Amplitude Normalization

</text>


</svg>
"""

    output_file = (
        OUTPUT_FOLDER
        /
        (
            class_name
            +
            "_original_vs_preprocessed.svg"
        )
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(svg)

    return output_file


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("NEONATAL CRY DATASET PREPROCESSING")
print("=" * 70)

print()
print("No external Python modules are required.")
print("Original audio files will NOT be modified.")
print()


# ============================================================
# CSV
# ============================================================

csv_file = (
    OUTPUT_FOLDER
    /
    "preprocessed_dataset.csv"
)


rows = []

total_files = 0
valid_files = 0
invalid_files = 0


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in CLASSES:

    folder = DATASET_FOLDER / class_name

    print("-" * 70)
    print("CLASS:", class_name)

    if not folder.exists():

        print(
            "WARNING: Folder not found"
        )

        continue

    wav_files = sorted(
        folder.glob("*.wav")
    )

    print(
        "WAV files found:",
        len(wav_files)
    )

    graph_created = False

    for wav_file in wav_files:

        total_files += 1

        try:

            # ------------------------------------------------
            # Read actual audio
            # ------------------------------------------------

            original, sample_rate, channels, bits = read_wav(
                wav_file
            )

            if len(original) == 0:

                raise ValueError(
                    "Empty audio file"
                )

            # ------------------------------------------------
            # Original measurements
            # ------------------------------------------------

            original_duration = (
                len(original)
                / sample_rate
            )

            original_rms = calculate_rms(
                original
            )

            original_zcr = calculate_zcr(
                original
            )

            # ------------------------------------------------
            # PREPROCESSING
            # ------------------------------------------------

            # Resample
            processed = resample(
                original,
                sample_rate,
                TARGET_SAMPLE_RATE
            )

            # Normalize
            processed = normalize(
                processed
            )

            # ------------------------------------------------
            # Processed measurements
            # ------------------------------------------------

            processed_duration = (
                len(processed)
                / TARGET_SAMPLE_RATE
            )

            processed_rms = calculate_rms(
                processed
            )

            processed_zcr = calculate_zcr(
                processed
            )

            # ------------------------------------------------
            # Add row to CSV
            # ------------------------------------------------

            rows.append([
                wav_file.name,
                class_name,
                sample_rate,
                TARGET_SAMPLE_RATE,
                channels,
                bits,
                round(
                    original_duration,
                    4
                ),
                round(
                    processed_duration,
                    4
                ),
                len(original),
                len(processed),
                round(
                    original_rms,
                    6
                ),
                round(
                    processed_rms,
                    6
                ),
                round(
                    original_zcr,
                    6
                ),
                round(
                    processed_zcr,
                    6
                ),
                "Valid"
            ])

            valid_files += 1

            # ------------------------------------------------
            # Create graph from FIRST VALID REAL FILE
            # OF THIS CLASS
            # ------------------------------------------------

            if not graph_created:

                graph_file = create_svg(
                    original,
                    processed,
                    class_name,
                    wav_file.name,
                    sample_rate
                )

                print()
                print(
                    "Graph created from actual file:"
                )

                print(
                    " ",
                    wav_file
                )

                print(
                    "Saved:",
                    graph_file
                )

                graph_created = True

        except Exception as error:

            invalid_files += 1

            rows.append([
                wav_file.name,
                class_name,
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "",
                "Invalid: " + str(error)
            ])

            print(
                "Invalid:",
                wav_file.name,
                "->",
                str(error)
            )


# ============================================================
# SAVE CSV
# ============================================================

with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "File_Name",
        "Class",
        "Original_Sample_Rate_Hz",
        "Processed_Sample_Rate_Hz",
        "Channels",
        "Bits_Per_Sample",
        "Original_Duration_sec",
        "Processed_Duration_sec",
        "Original_Samples",
        "Processed_Samples",
        "Original_RMS",
        "Processed_RMS",
        "Original_ZCR",
        "Processed_ZCR",
        "Status"
    ])

    writer.writerows(rows)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print()
print("Total WAV files :", total_files)
print("Valid files     :", valid_files)
print("Invalid files   :", invalid_files)

print()
print("CSV:")
print(
    " ",
    csv_file
)

print()
print("Graphs:")
print(
    " ",
    OUTPUT_FOLDER
)

print()
print("Created files are based on REAL recordings")
print("from your dataset.")
print()
print("No original WAV file was changed.")
print("=" * 70)
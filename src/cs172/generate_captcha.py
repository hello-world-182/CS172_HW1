"""Synthetic CAPTCHA generation used for the training and validation sets."""

from __future__ import annotations

import random
import colorsys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=32)
def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.truetype("arial.ttf", size=size)


_FONT_DIRS = (
    Path("C:/Windows/Fonts"),
    Path("/usr/share/fonts/truetype/msttcorefonts"),
    Path("/usr/share/fonts/truetype/liberation2"),
    Path("/usr/share/fonts/truetype/liberation"),
    Path("/usr/share/fonts/truetype/dejavu"),
)
# Sans-serif families with the same digit shapes as the reference images
# (single-stroke "1" without a foot, closed "4"). Arial dominates; the rest
# add mild shape variety so the model does not depend on one rasterization.
_MATCHED_FONTS = (
    ("arial.ttf", 0.70),
    ("LiberationSans-Regular.ttf", 0.10),
    ("calibri.ttf", 0.05),
    ("segoeui.ttf", 0.05),
    ("tahoma.ttf", 0.05),
    ("verdana.ttf", 0.05),
)


@lru_cache(maxsize=1)
def _matched_font_files() -> tuple[tuple[str, float], ...]:
    found = []
    for name, weight in _MATCHED_FONTS:
        for directory in _FONT_DIRS:
            if not directory.is_dir():
                continue
            matches = [path for path in directory.iterdir() if path.name.lower() == name.lower()]
            if matches:
                found.append((str(matches[0]), weight))
                break
    if not found:
        fallback = _font(30)
        found.append((fallback.path, 1.0))
    return tuple(found)


@lru_cache(maxsize=256)
def _font_file(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


def _render_glyph(character, font, color, angle):
    """Render one character on a transparent canvas, rotate, and crop to its ink."""
    bounds = font.getbbox(character)
    glyph = Image.new("RGBA", (bounds[2] - bounds[0] + 12, bounds[3] - bounds[1] + 12), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text((6 - bounds[0], 6 - bounds[1]), character, font=font, fill=color)
    if angle:
        glyph = glyph.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
    box = glyph.getchannel("A").getbbox()
    return glyph.crop(box) if box else glyph


def _matched_color(rng: random.Random, grayscale: bool) -> tuple[int, int, int, int]:
    if grayscale:
        shade = rng.randint(0, 30)
        return (shade, shade, shade, 255)
    hue = rng.random()
    saturation = rng.uniform(0.25, 0.95)
    value = rng.uniform(0.30, 0.85)
    return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(hue, saturation, value)) + (255,)


def _generate_matched(width, height, num_digits, font_size_range, max_noisy_lines, noise_ratio, border_width, seed):
    """Profile modeled on the provided sample appearance.

    The reference folders show five families: black clean digits, colored
    clean digits, colored digits with pixel noise, noise plus interference
    lines, and the same with strong rotation. In all of them the digits use
    the full canvas height, gaps between digits are very uneven (digits may
    almost touch), and images are JPEG compressed. Families are sampled
    uniformly and each factor is also toggled independently some of the time
    so the model does not rely on the factors appearing together.
    """
    rng = random.Random(seed)
    family = rng.randrange(5)
    grayscale = family == 0
    use_noise = family >= 2
    use_lines = family >= 3
    strong_rotation = family == 4
    if rng.random() < 0.25:
        grayscale = rng.random() < 0.25
        use_noise = rng.random() < 0.6
        use_lines = rng.random() < 0.5
        strong_rotation = rng.random() < 0.4
    angle_limit = 30.0 if strong_rotation else rng.choice((0.0, 0.0, 5.0, 12.0))

    characters = [str(rng.randrange(10)) for _ in range(num_digits)]
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def draw_lines():
        for _ in range(rng.randint(1, max(1, max_noisy_lines))):
            color = tuple(rng.randint(40, 230) for _ in range(3))
            draw.line(
                (rng.randrange(width), rng.randrange(height), rng.randrange(width), rng.randrange(height)),
                fill=color,
                width=rng.choice((1, 1, 1, 2)),
            )

    lines_on_top = rng.random() < 0.5
    if use_lines and max_noisy_lines and not lines_on_top:
        draw_lines()

    font_files = _matched_font_files()
    font_path = rng.choices([path for path, _ in font_files], weights=[w for _, w in font_files])[0]
    glyphs = []
    for character in characters:
        font = _font_file(font_path, rng.randint(*font_size_range))
        angle = rng.uniform(-angle_limit, angle_limit) if angle_limit else 0.0
        glyphs.append(_render_glyph(character, font, _matched_color(rng, grayscale), angle))

    inner = border_width + 1
    left, right = inner + rng.randint(2, 18), width - inner - rng.randint(2, 18)
    total = sum(glyph.width for glyph in glyphs)
    free = right - left - total
    if free < 0:
        left, right = inner, width - inner
        free = max(0, right - left - total)
    # Small Dirichlet concentration gives very uneven gaps; larger gives
    # roughly even spacing. Mixing both covers the observed layouts.
    shape = rng.choice((0.4, 0.8, 1.5, 4.0))
    weights = [rng.gammavariate(shape, 1.0) for _ in range(num_digits - 1)]
    weight_total = sum(weights) or 1.0
    gaps = [free * w / weight_total for w in weights] + [0.0]

    shared_baseline = rng.random() < 0.3
    base_center = rng.uniform(height * 0.35, height * 0.65)
    x = float(left)
    for glyph, gap in zip(glyphs, gaps):
        min_y, max_y = inner, max(inner, height - inner - glyph.height)
        if shared_baseline:
            y = int(round(base_center - glyph.height / 2 + rng.uniform(-6, 6)))
        else:
            y = rng.randint(min_y, max_y)
        y = max(min_y, min(max_y, y))
        px = max(inner, min(width - inner - glyph.width, int(round(x))))
        image.paste(glyph, (px, y), glyph)
        x += glyph.width + gap

    if use_lines and max_noisy_lines and lines_on_top:
        draw_lines()

    if use_noise and noise_ratio:
        noise_rng = np.random.default_rng(seed)
        ratio = noise_ratio * rng.uniform(0.6, 1.4)
        count = round(width * height * ratio)
        xs = noise_rng.integers(0, width, size=count)
        ys = noise_rng.integers(0, height, size=count)
        pixels = np.asarray(image).copy()
        pixels[ys, xs] = noise_rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")

    if border_width:
        ImageDraw.Draw(image).rectangle((0, 0, width - 1, height - 1), outline=(0, 0, 0), width=border_width)
    return image, characters


@lru_cache(maxsize=64)
def _default_font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.load_default(size=size)


def _generate_reference(width, height, num_digits, font_size_range, minimum_padding, max_noisy_lines,
                        noise_ratio, border_width, seed):
    """Profile modeled on the appearance of the provided CAPTCHA examples.

    The reference glyphs match Pillow's built-in default font (Aileron) at
    30-40 px. Ink stays inside the horizontal/vertical padding, digits are
    laid out left to right with at least a few pixels between them, and the
    five difficulty levels add colour, 2% pixel noise, up to five lines, and
    +/-30 degree rotation in that order. A minority of samples use other sans
    fonts and wider placement so the model does not rely on exact pixels.
    """
    rng = random.Random(seed)
    level = rng.randrange(5)
    color = level >= 1
    use_noise = level >= 2
    use_lines = level >= 3
    rotate = level >= 4
    if rng.random() < 0.2:
        color = rng.random() < 0.8
        use_noise = rng.random() < 0.6
        use_lines = rng.random() < 0.5
        rotate = rng.random() < 0.4
    other_font = rng.random() < 0.1
    loose = rng.random() < 0.15

    characters = [str(rng.randrange(10)) for _ in range(num_digits)]
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    def draw_lines():
        for _ in range(rng.randint(1, max(1, max_noisy_lines))):
            draw.line(
                (rng.randrange(width), rng.randrange(height), rng.randrange(width), rng.randrange(height)),
                fill=tuple(rng.randint(0, 255) for _ in range(3)),
                width=rng.choice((1, 1, 1, 2)),
            )

    lines_on_top = rng.random() < 0.7
    if use_lines and max_noisy_lines and not lines_on_top:
        draw_lines()

    font_files = _matched_font_files()
    glyphs = []
    for character in characters:
        size = rng.randint(*font_size_range)
        if other_font:
            font = _font_file(rng.choice(font_files)[0], size)
        else:
            font = _default_font(size)
        if color:
            fill = tuple(rng.randint(0, 200) for _ in range(3)) + (255,)
        else:
            shade = rng.randint(0, 20) if loose else 0
            fill = (shade, shade, shade, 255)
        angle = rng.uniform(-30, 30) if rotate else 0.0
        glyphs.append(_render_glyph(character, font, fill, angle))

    pad_x, pad_y = minimum_padding
    inner = border_width + 1
    left, right = (inner, width - inner) if loose else (pad_x, width - pad_x)
    min_gap = rng.randint(0, 3) if loose or rotate else rng.randint(3, 6)
    total = sum(glyph.width for glyph in glyphs) + min_gap * (num_digits - 1)
    if total > right - left:
        left, right = inner, width - inner
        min_gap = max(0, (right - left - sum(glyph.width for glyph in glyphs)) // (num_digits - 1))
        total = sum(glyph.width for glyph in glyphs) + min_gap * (num_digits - 1)
    free = max(0, right - left - total)
    weights = [rng.gammavariate(rng.choice((0.7, 1.0, 1.0, 1.5)), 1.0) for _ in range(num_digits + 1)]
    weight_total = sum(weights) or 1.0
    gaps = [free * w / weight_total for w in weights]

    # Reference ink tops start about 2*pad_y below the edge because the
    # default font has ~10 px of internal leading; bottoms reach the border.
    top_low = inner if (loose or rotate) else 2 * pad_y - rng.randint(0, 4)
    x = left + gaps[0]
    for glyph, gap in zip(glyphs, gaps[1:]):
        y_max = max(inner, height - inner - glyph.height)
        y = rng.randint(min(top_low, y_max), y_max)
        px = max(inner, min(width - inner - glyph.width, int(round(x))))
        image.paste(glyph, (px, y), glyph)
        x += glyph.width + min_gap + gap

    if use_lines and max_noisy_lines and lines_on_top:
        draw_lines()

    if use_noise and noise_ratio:
        noise_rng = np.random.default_rng(seed)
        count = round(width * height * noise_ratio * rng.uniform(0.8, 1.2))
        xs = noise_rng.integers(0, width, size=count)
        ys = noise_rng.integers(0, height, size=count)
        pixels = np.asarray(image).copy()
        pixels[ys, xs] = noise_rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")

    if border_width:
        ImageDraw.Draw(image).rectangle((0, 0, width - 1, height - 1), outline=(0, 0, 0), width=border_width)
    return image, characters


def generate_captcha(
    image_size: tuple[int, int] = (260, 80),
    num_digits: int = 5,
    font_size_range: tuple[int, int] = (30, 40),
    minimum_padding: tuple[int, int] = (5, 10),
    rotation_angle_range: tuple[int, int] | None = (-30, 30),
    max_noisy_lines: int = 5,
    noise_ratio: float = 0.02,
    border_width: int = 2,
    domain_randomization: bool = False,
    test_like: bool = False,
    matched: bool = False,
    reference: bool = False,
    seed: int | None = None,
) -> tuple[Image.Image, list[str]]:
    """Create an RGB CAPTCHA and its ordered digits using a local seeded RNG."""
    width, height = image_size
    if width < 1 or height < 1 or num_digits < 1:
        raise ValueError("image dimensions and num_digits must be positive")
    if font_size_range[0] < 1 or font_size_range[0] > font_size_range[1]:
        raise ValueError("font_size_range must be positive and ascending")
    if max_noisy_lines < 0 or not 0 <= noise_ratio <= 1 or border_width < 0:
        raise ValueError("invalid line, noise, or border setting")
    if minimum_padding[0] * 2 >= width or minimum_padding[1] * 2 >= height:
        raise ValueError("minimum padding leaves no room for the CAPTCHA")

    if reference:
        return _generate_reference(
            width, height, num_digits, font_size_range, minimum_padding, max_noisy_lines, noise_ratio,
            border_width, seed,
        )
    if matched:
        return _generate_matched(
            width, height, num_digits, font_size_range, max_noisy_lines, noise_ratio, border_width, seed
        )

    rng = random.Random(seed)
    if test_like:
        domain_randomization = True
    style = rng.randrange(5) if domain_randomization else None
    image_noise_ratio = noise_ratio
    line_count_range = (0, max_noisy_lines)
    angle_range = rotation_angle_range
    grayscale_only = False
    vertical_shift = 0
    gap_shape = 2.0
    vertical_spread = 0.0
    black_border = False
    if style is not None:
        # Five broad rendering domains cover clean monochrome, clean color,
        # pixel noise, and increasing interference while keeping each image
        # fully synthetic and reproducible from its seed.
        image_noise_ratio = 0.0 if style < 2 else noise_ratio
        line_count_range = ((0, 0), (0, 0), (0, 0), (1, min(3, max_noisy_lines)), (1, max_noisy_lines))[style]
        if max_noisy_lines == 0:
            line_count_range = (0, 0)
        if rotation_angle_range is not None:
            angle_range = ((-30, 30), (-30, 30), (-30, 30), (-30, 30), (-30, 30))[style] if test_like else ((-10, 10), (-10, 10), (-18, 18), (-30, 30), (-30, 30))[style]
        grayscale_only = style == 0
        vertical_shift = 10
        gap_shape = 0.7
        vertical_spread = 11.0 if test_like else 0.0
        black_border = test_like

    characters = [str(rng.randrange(10)) for _ in range(num_digits)]
    background = (255, 255, 255) if style is not None else (248, 248, 244)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(*line_count_range)):
        color = tuple(rng.randint(80 if test_like else 110, 220) for _ in range(3))
        draw.line(
            (rng.randrange(width), rng.randrange(height), rng.randrange(width), rng.randrange(height)),
            fill=color,
            width=rng.choice((1, 1, 2)),
        )

    usable_left, usable_right = minimum_padding[0], width - minimum_padding[0]
    slot_width = (usable_right - usable_left) / num_digits
    glyphs = []
    for index, character in enumerate(characters):
        font = _font(rng.randint(*font_size_range))
        bounds = font.getbbox(character, stroke_width=0)
        glyph_width = max(1, bounds[2] - bounds[0])
        glyph_height = max(1, bounds[3] - bounds[1])
        glyph = Image.new("RGBA", (glyph_width + 8, glyph_height + 8), (0, 0, 0, 0))
        glyph_draw = ImageDraw.Draw(glyph)
        if grayscale_only or rng.random() < 0.2:
            shade = 0 if test_like and grayscale_only else rng.randint(5, 40)
            color = (shade, shade, shade, 255)
        else:
            hue = rng.random()
            saturation = rng.uniform(0.30, 0.95) if style is not None else rng.uniform(0.55, 0.95)
            brightness = rng.uniform(0.25, 0.78) if test_like else (rng.uniform(0.35, 0.88) if style is not None else rng.uniform(0.35, 0.78))
            color = tuple(round(channel * 255) for channel in colorsys.hsv_to_rgb(hue, saturation, brightness)) + (255,)
        glyph_draw.text((4 - bounds[0], 4 - bounds[1]), character, font=font, fill=color)
        if angle_range is not None:
            glyph = glyph.rotate(rng.uniform(*angle_range), resample=Image.Resampling.BICUBIC, expand=True)
        glyphs.append(glyph)

    if style is not None:
        # Vary inter-character spacing without changing digit order or letting
        # glyphs overlap. A random gap allocation avoids the fixed five-slot
        # geometry used by the baseline generator.
        free_space = max(0, usable_right - usable_left - sum(glyph.width for glyph in glyphs))
        gap_weights = [rng.gammavariate(gap_shape, 1.0) for _ in range(num_digits + 1)]
        weight_total = sum(gap_weights)
        gaps = [free_space * weight / weight_total for weight in gap_weights]
        x = usable_left + gaps[0]
        for glyph, gap in zip(glyphs, gaps[1:]):
            min_y = minimum_padding[1] + vertical_shift
            max_y = max(min_y, min(height - glyph.height - border_width, height - minimum_padding[1] - glyph.height + vertical_shift))
            if vertical_spread:
                vertical_center = 27.0 if style == 4 else 38.0
                spread = 12.0 if style == 4 else vertical_spread
                y = int(round(max(min_y, min(max_y, vertical_center + rng.gauss(0.0, spread)))))
            else:
                y = rng.randint(min_y, max_y) if max_y > min_y else min_y
            image.paste(glyph, (round(x), y), glyph)
            x += glyph.width + gap
    else:
        for index, glyph in enumerate(glyphs):
            center_x = usable_left + (index + 0.5) * slot_width + rng.uniform(-slot_width * 0.12, slot_width * 0.12)
            min_y = minimum_padding[1] + vertical_shift
            max_y = max(min_y, min(height - glyph.height - border_width, height - minimum_padding[1] - glyph.height + vertical_shift))
            if vertical_spread:
                vertical_center = 27.0 if style == 4 else 38.0
                spread = 12.0 if style == 4 else vertical_spread
                y = int(round(max(min_y, min(max_y, vertical_center + rng.gauss(0.0, spread)))))
            else:
                y = int(rng.randint(min_y, max_y)) if max_y > min_y else min_y
            x = max(0, min(width - glyph.width, int(center_x - glyph.width / 2)))
            image.paste(glyph, (x, y), glyph)

    if image_noise_ratio:
        noise_rng = np.random.default_rng(seed)
        count = round(width * height * image_noise_ratio)
        x = noise_rng.integers(0, width, size=count)
        y = noise_rng.integers(0, height, size=count)
        pixels = np.asarray(image).copy()
        pixels[y, x] = noise_rng.integers(0, 256, size=(count, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, mode="RGB")

    if border_width:
        ImageDraw.Draw(image).rectangle(
            (0, 0, width - 1, height - 1), outline=(0, 0, 0) if black_border else (45, 45, 45), width=border_width
        )
    return image, characters

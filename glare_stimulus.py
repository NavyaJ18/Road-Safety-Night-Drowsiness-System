#!/usr/bin/env python3
"""
Disc-and-annulus stimulus for the headlight-glare threshold experiment.

Modes
-----
  python glare_stimulus.py calibrate   Step the disc through grayscale values so you can
                                       read each one with the luminance meter (Step 5).
  python glare_stimulus.py run         Run the threshold trials (method of adjustment)
                                       and log the grayscale step at threshold (Step 10).

  python glare_stimulus.py adjust      Free play: change the disc brightness at will
                                       (UP/DOWN = +/-1, SHIFT+UP/DOWN = +/-5).

Add --windowed to test on any screen without going full-screen.

Install:  pip install pygame

Keys (run mode)
---------------
  SPACE        start the next trial (disc starts invisible)
  UP / W       disc +1 grayscale step      (one step per key press)
  DOWN / S     disc -1 grayscale step
  ENTER        record the current disc step as the threshold
  ESC          quit

Keys (calibrate mode)
---------------------
  RIGHT / SPACE   next grayscale step      LEFT   previous step      ESC   quit
"""
import argparse
import csv
import math
import os
import random
import sys
import time

import pygame

# ============================== CONFIGURATION ==============================
# --- Display geometry (measure these!) ---
SCREEN_WIDTH_CM = 34.5      # physical width of the visible screen area, in cm
VIEW_DIST_CM = 100.0        # eye-to-screen distance, in cm

# --- Stimulus size: same visual angle as the paper (1.60 cm disc and 5.60 cm
#     annulus outer diameter, viewed from 77.5 cm) ---
PAPER_DIST_CM = 77.5
PAPER_DISC_CM = 1.60
PAPER_ANNULUS_CM = 5.60

# --- Fixed physical sizes ON THE SCREEN (cm). Set to None to use the visual-angle
#     scaling above. With 1.60 / 5.60 the sizes are exact on screen at any distance
#     (the visual angle then matches the paper only if you sit at 77.5 cm). ---
FIXED_DISC_CM = 1.60
FIXED_ANNULUS_CM = 5.60

# --- Grayscale values (0-255). Keep fixed for the whole study and record the
#     luminance of each with the luminance meter. ---
BACKGROUND_STEP = 20        # field around the annulus (choose a dim gray)
ANNULUS_STEP = 0            # annulus (black); its luminance is L_B
START_STEP = ANNULUS_STEP   # disc starts here, i.e. invisible against the annulus

# --- Trial design (replace dial values with your measured, calibrated ones) ---
POSITIONS = {"P1": 13, "P2": 20, "P3": 27}   # lamp position -> eccentricity (deg)
DIAL_LEVELS = [3, 9, 31, 93]                 # lamp dial settings (%)
BASELINE_REPEATS = 3                         # lamp-off threshold trials
LOG_FILE = "threshold_log.csv"

# --- Keys ---
UP_KEYS = (pygame.K_UP, pygame.K_w)
DOWN_KEYS = (pygame.K_DOWN, pygame.K_s)
RECORD_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER)
NEXT_KEYS = (pygame.K_RIGHT, pygame.K_SPACE)
PREV_KEYS = (pygame.K_LEFT,)

# Grayscale steps to read with the luminance meter: dense at the dark end.
CALIBRATION_STEPS = (
    list(range(0, 11))
    + list(range(15, 31, 5))
    + list(range(35, 91, 5))
    + [100, 120, 150, 175, 200, 225, 250, 255]
)
# ===========================================================================


# ------------------------------ geometry ----------------------------------
def visual_angle_deg(size_cm, dist_cm):
    return 2 * math.degrees(math.atan((size_cm / 2) / dist_cm))


def size_cm_for_angle(angle_deg, dist_cm):
    return 2 * dist_cm * math.tan(math.radians(angle_deg / 2))


def compute_geometry(screen_width_px):
    """Return disc/annulus radii in pixels for the configured viewing distance."""
    px_per_cm = screen_width_px / SCREEN_WIDTH_CM
    disc_deg = visual_angle_deg(PAPER_DISC_CM, PAPER_DIST_CM)
    annulus_deg = visual_angle_deg(PAPER_ANNULUS_CM, PAPER_DIST_CM)
    disc_cm = size_cm_for_angle(disc_deg, VIEW_DIST_CM)
    annulus_cm = size_cm_for_angle(annulus_deg, VIEW_DIST_CM)
    if FIXED_DISC_CM is not None:          # force the disc to a physical size on screen
        disc_cm = FIXED_DISC_CM
        disc_deg = visual_angle_deg(disc_cm, VIEW_DIST_CM)
    if FIXED_ANNULUS_CM is not None:
        annulus_cm = FIXED_ANNULUS_CM
        annulus_deg = visual_angle_deg(annulus_cm, VIEW_DIST_CM)
    return {
        "px_per_cm": px_per_cm,
        "disc_deg": disc_deg,
        "annulus_deg": annulus_deg,
        "disc_cm": disc_cm,
        "annulus_cm": annulus_cm,
        "disc_r": max(1, round(disc_cm * px_per_cm / 2)),
        "annulus_r": max(2, round(annulus_cm * px_per_cm / 2)),
    }


def gray(step):
    step = max(0, min(255, int(step)))
    return (step, step, step)


# ------------------------------ drawing -----------------------------------
def draw_stimulus(surface, disc_step, geom):
    """Gray background, black annulus, and a disc whose gray step you control.

    The annulus is the dark ring between the disc edge and the outer circle, so the
    disc is seen against the annulus. Plain (non-antialiased) circles are used so
    that every pixel has exactly the requested grayscale value.
    """
    w, h = surface.get_size()
    centre = (w // 2, h // 2)
    surface.fill(gray(BACKGROUND_STEP))
    pygame.draw.circle(surface, gray(ANNULUS_STEP), centre, geom["annulus_r"])
    pygame.draw.circle(surface, gray(disc_step), centre, geom["disc_r"])


# ------------------------------ setup -------------------------------------
def enable_dpi_awareness():
    """On Windows, stop display scaling from changing the pixel size."""
    if sys.platform.startswith("win"):
        try:
            import ctypes
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def open_display(windowed):
    enable_dpi_awareness()
    pygame.init()
    if windowed:
        screen = pygame.display.set_mode((1280, 720))
    else:
        screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    pygame.mouse.set_visible(False)
    pygame.display.set_caption("Glare threshold stimulus")
    return screen


def print_geometry(geom, screen):
    w, h = screen.get_size()
    print(f"Screen: {w} x {h} px, {SCREEN_WIDTH_CM} cm wide -> {geom['px_per_cm']:.2f} px/cm")
    print(f"Viewing distance {VIEW_DIST_CM:.0f} cm")
    print(f"Disc:    {geom['disc_deg']:.2f} deg = {geom['disc_cm']:.2f} cm = "
          f"{2 * geom['disc_r']} px diameter")
    print(f"Annulus: {geom['annulus_deg']:.2f} deg = {geom['annulus_cm']:.2f} cm = "
          f"{2 * geom['annulus_r']} px outer diameter")
    print("Check both with a ruler held on the screen.\n")


# ------------------------------ calibrate mode ----------------------------
def run_calibration(screen):
    geom = compute_geometry(screen.get_width())
    print_geometry(geom, screen)
    print("CALIBRATE: aim the luminance meter at the disc and read each step.")
    print("RIGHT/SPACE = next step, LEFT = previous, ESC = quit.\n")
    clock = pygame.time.Clock()
    idx = 0
    changed = True
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in NEXT_KEYS and idx < len(CALIBRATION_STEPS) - 1:
                    idx += 1
                    changed = True
                elif event.key in PREV_KEYS and idx > 0:
                    idx -= 1
                    changed = True
        if changed:
            step = CALIBRATION_STEPS[idx]
            print(f"Step {idx + 1}/{len(CALIBRATION_STEPS)}: disc grayscale = {step}")
            changed = False
        draw_stimulus(screen, CALIBRATION_STEPS[idx], geom)
        pygame.display.flip()
        clock.tick(60)
    print("\nAlso read once: the annulus (grayscale "
          f"{ANNULUS_STEP}) and the background (grayscale {BACKGROUND_STEP}).")


# ------------------------------ adjust mode -------------------------------
def run_adjust(screen):
    """Free play: the observer changes the disc brightness at will."""
    geom = compute_geometry(screen.get_width())
    print_geometry(geom, screen)
    print("ADJUST (disc starts WHITE so you can see it):")
    print("  Brightness: UP/W brighter, DOWN/S darker (SHIFT = +/-5), PAGEUP/PAGEDOWN = +/-20,")
    print("              0 = black (invisible), 9 = white")
    print("  Disc size:  RIGHT = bigger, LEFT = smaller (SHIFT = 0.5 cm steps, else 0.1 cm)")
    print("  Ring size:  D = bigger, A = smaller (SHIFT = 0.5 cm steps, else 0.1 cm)")
    print("  ESC = quit\n")
    clock = pygame.time.Clock()
    disc_step = 255
    disc_cm = geom["disc_cm"]
    ring_cm = geom["annulus_cm"]
    last_printed = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    continue
                shift = bool(event.mod & pygame.KMOD_SHIFT)
                big = 5 if shift else 1
                size = 0.5 if shift else 0.1
                if event.key in UP_KEYS:
                    disc_step = min(255, disc_step + big)
                elif event.key in DOWN_KEYS:
                    disc_step = max(0, disc_step - big)
                elif event.key == pygame.K_PAGEUP:
                    disc_step = min(255, disc_step + 20)
                elif event.key == pygame.K_PAGEDOWN:
                    disc_step = max(0, disc_step - 20)
                elif event.key in (pygame.K_0, pygame.K_KP0):
                    disc_step = 0
                elif event.key in (pygame.K_9, pygame.K_KP9):
                    disc_step = 255
                elif event.key == pygame.K_RIGHT:
                    disc_cm += size
                elif event.key == pygame.K_LEFT:
                    disc_cm = max(0.1, disc_cm - size)
                elif event.key == pygame.K_d:
                    ring_cm += size
                elif event.key == pygame.K_a:
                    ring_cm = max(0.2, ring_cm - size)
                disc_cm = min(disc_cm, ring_cm)      # disc always stays inside the ring
        geom["disc_cm"], geom["annulus_cm"] = disc_cm, ring_cm
        geom["disc_r"] = max(1, round(disc_cm * geom["px_per_cm"] / 2))
        geom["annulus_r"] = max(2, round(ring_cm * geom["px_per_cm"] / 2))
        state = (disc_step, round(disc_cm, 2), round(ring_cm, 2))
        if state != last_printed:
            print(f"Disc grayscale = {disc_step} | disc {disc_cm:.2f} cm "
                  f"({2 * geom['disc_r']} px) | ring {ring_cm:.2f} cm ({2 * geom['annulus_r']} px)")
            last_printed = state
        draw_stimulus(screen, disc_step, geom)
        pygame.display.flip()
        clock.tick(60)


# ------------------------------ run mode ----------------------------------
def build_schedule(seed):
    rng = random.Random(seed)
    glare = [
        {"position": pos, "theta_deg": theta, "dial_pct": dial}
        for pos, theta in POSITIONS.items()
        for dial in DIAL_LEVELS
    ]
    rng.shuffle(glare)
    baseline = [{"position": "OFF", "theta_deg": "", "dial_pct": 0}
                for _ in range(BASELINE_REPEATS)]
    return baseline + glare


def describe(trial):
    if trial["position"] == "OFF":
        return "lamp OFF (baseline)"
    return f"lamp at {trial['position']} ({trial['theta_deg']} deg), dial {trial['dial_pct']}%"


def append_log(row):
    new_file = not os.path.exists(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_trials(screen, observer_id, age, seed):
    geom = compute_geometry(screen.get_width())
    print_geometry(geom, screen)
    schedule = build_schedule(seed)
    print(f"Observer {observer_id}, age {age}. Trial order (seed {seed}):")
    for i, trial in enumerate(schedule, 1):
        print(f"  {i:2d}. {describe(trial)}")
    print("\nSet the lamp for trial 1, dark-adapt the observer, then press SPACE.\n")

    clock = pygame.time.Clock()
    trial_idx = 0
    state = "WAIT"          # WAIT = disc hidden; ADJUST = observer is adjusting
    disc_step = START_STEP
    presses = 0
    t_start = 0.0
    running = True

    while running and trial_idx < len(schedule):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif state == "WAIT" and event.key == pygame.K_SPACE:
                    state = "ADJUST"
                    disc_step = START_STEP
                    presses = 0
                    t_start = time.time()
                elif state == "ADJUST":
                    if event.key in UP_KEYS:
                        disc_step = min(255, disc_step + 1)
                        presses += 1
                    elif event.key in DOWN_KEYS:
                        disc_step = max(0, disc_step - 1)
                        presses += 1
                    elif event.key in RECORD_KEYS:
                        trial = schedule[trial_idx]
                        append_log({
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "observer_id": observer_id,
                            "age": age,
                            "trial_no": trial_idx + 1,
                            "position": trial["position"],
                            "theta_deg": trial["theta_deg"],
                            "dial_pct": trial["dial_pct"],
                            "disc_step": disc_step,
                            "annulus_step": ANNULUS_STEP,
                            "background_step": BACKGROUND_STEP,
                            "n_presses": presses,
                            "duration_s": round(time.time() - t_start, 1),
                        })
                        print(f"Trial {trial_idx + 1}/{len(schedule)} recorded: "
                              f"disc step = {disc_step}")
                        trial_idx += 1
                        state = "WAIT"
                        disc_step = START_STEP
                        if trial_idx < len(schedule):
                            print(f"  Next: {describe(schedule[trial_idx])}. "
                                  "Rest the observer 1-2 min, set the lamp, press SPACE.")
        draw_stimulus(screen, disc_step if state == "ADJUST" else START_STEP, geom)
        pygame.display.flip()
        clock.tick(60)

    if trial_idx >= len(schedule):
        print(f"\nAll trials done. Results saved to {LOG_FILE}.")
    else:
        print(f"\nStopped early after {trial_idx} trials. Results saved to {LOG_FILE}.")


# ------------------------------ main --------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["calibrate", "run", "adjust"])
    parser.add_argument("--windowed", action="store_true", help="do not use full-screen")
    parser.add_argument("--seed", type=int, default=None,
                        help="random seed for the trial order (default: random)")
    args = parser.parse_args(argv)

    observer_id, age = "", ""
    seed = args.seed if args.seed is not None else random.randrange(10**6)
    if args.mode == "run":
        observer_id = input("Observer ID: ").strip()
        age = input("Observer age (years): ").strip()

    screen = open_display(args.windowed)
    try:
        if args.mode == "calibrate":
            run_calibration(screen)
        elif args.mode == "adjust":
            run_adjust(screen)
        else:
            run_trials(screen, observer_id, age, seed)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()
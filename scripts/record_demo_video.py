import os
import shutil
import subprocess
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

def main():
    output_dir = Path("demo_output")
    output_dir.mkdir(exist_ok=True)
    raw_video_dir = output_dir / "raw_recordings"
    if raw_video_dir.exists():
        shutil.rmtree(raw_video_dir)
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    final_mp4 = output_dir / "rosetta_shard_demo.mp4"
    final_webm = output_dir / "rosetta_shard_demo.webm"

    print("Starting Playwright with Edge browser...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            channel="msedge",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )

        # 1440x900 viewport gives spacious 16:10 presentation where the control center fits completely
        context = browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(raw_video_dir),
            record_video_size={"width": 1440, "height": 900}
        )

        page = context.new_page()

        print("Navigating to Rosetta Shard server at http://127.0.0.1:8000...")
        page.goto("http://127.0.0.1:8000", wait_until="networkidle")
        page.wait_for_timeout(1000)

        # Reset state to clean baseline
        print("Resetting demo state...")
        page.evaluate("resetDemo()")
        page.wait_for_timeout(1500)

        # 1. Landing Hero Showcase
        print("Showcasing landing header...")
        page.wait_for_timeout(2500)

        # Smooth scroll down to Demo section and align at top
        print("Scrolling smoothly to Live Product Demo section...")
        page.evaluate("window.scrollTo({top: document.getElementById('demo').offsetTop - 20, behavior: 'smooth'})")
        page.wait_for_timeout(2500)

        # 2. STEP A: Airplane Mode Query
        print("Step A: Enabling Airplane Mode...")
        btn_airplane = page.locator("#btn-airplane")
        btn_airplane.click()
        page.wait_for_timeout(1500)

        print("Step A: Executing local on-device semantic query in Airplane Mode...")
        btn_search = page.locator("button:has-text('Search')")
        btn_search.click()
        page.wait_for_timeout(3500)

        # 3. STEP B: Cloud Model Upgrade & Space Guard
        print("Step B: Disabling Airplane Mode...")
        btn_airplane.click()
        page.wait_for_timeout(1500)

        print("Step B: Upgrading Cloud Model to bge-large (1024-d)...")
        page.locator("#btn-upgrade").click()
        page.wait_for_timeout(3000)

        print("Step B: Temporarily Bypassing Space Guard to demonstrate silent mismatch risk...")
        btn_guard = page.locator("#btn-guard")
        btn_guard.click()
        page.wait_for_timeout(1500)

        print("Step B: Submitting stale query with Space Guard bypassed...")
        btn_search.click()
        page.wait_for_timeout(3000)

        print("Step B: Re-activating Space Guard protection...")
        btn_guard.click()
        page.wait_for_timeout(1500)

        print("Step B: Submitting query with Space Guard ACTIVE (expecting blocked mismatch)...")
        btn_search.click()
        page.wait_for_timeout(3500)

        # 4. STEP C: Push Compact Adapter & Atomic Shard Migration
        print("Step C: Pushing Compact Adapter Bundle (128 KB)...")
        page.locator("#btn-adapter").click()
        page.wait_for_timeout(3000)

        print("Step C: Executing Atomic Shard Migration...")
        page.locator("#btn-migrate").click()
        page.wait_for_timeout(4000)

        # 5. STEP D: In-Domain vs OOD Queries
        print("Step D: Running In-Domain Query (Robotics SLAM)...")
        page.locator(".preset-chip:has-text('Robotics SLAM')").click()
        page.wait_for_timeout(3500)

        print("Step D: Running Out-Of-Domain Shift Query (Crypto / Tax Policy)...")
        page.locator(".preset-chip:has-text('OOD Shift')").click()
        page.wait_for_timeout(4500)

        print("Finished demo walkthrough. Closing page and saving video stream...")
        page.wait_for_timeout(2000)
        video_path = page.video.path()
        context.close()
        browser.close()

    print(f"Raw recorded video saved to: {video_path}")

    # Transcode using system ffmpeg for high compatibility MP4 and WebM
    print("Transcoding video to MP4 and WebM using ffmpeg...")
    raw_vid = Path(video_path)
    if raw_vid.exists():
        shutil.copyfile(raw_vid, final_webm)
        print(f"WebM saved to: {final_webm}")

        # Convert to MP4
        cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_vid),
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(final_mp4)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Pristine MP4 successfully generated: {final_mp4}")
        else:
            print("FFmpeg transcoding warning:", result.stderr)

    print("Demo recording complete!")

if __name__ == "__main__":
    main()

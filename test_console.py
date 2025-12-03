#!/usr/bin/env python3
"""Test script for console substep functionality."""

import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import only the console module to avoid dependency issues
from targon.core.console import console


def main():
    with console(app_name="Test App") as c:
        # Test basic step and success
        c.step("Initializing application")
        time.sleep(0.5)
        c.success("Application initialized")
        
        # Test step with multiple substeps
        c.step("Building container image")
        time.sleep(0.3)
        c.substep("Downloading base image")
        time.sleep(0.3)
        c.substep("Installing dependencies - this is a really long message that should be truncated if it exceeds terminal width instead of wrapping to multiple lines")
        time.sleep(0.3)
        c.substep("Copying source code")
        time.sleep(0.3)
        c.substep("Step 1/5: Building layer 1")
        time.sleep(0.2)
        c.substep("Step 2/5: Building layer 2")
        time.sleep(0.2)
        c.substep("Step 3/5: Building layer 3")
        time.sleep(0.2)
        c.substep("Step 4/5: Building layer 4")
        time.sleep(0.2)
        c.substep("Step 5/5: Building layer 5")
        time.sleep(0.3)
        c.success("Container image built", duration=2.5)
        
        # Test resource display
        c.resource("Image", "sha256:abc123def456")
        
        # Test step with substeps and detail
        c.step("Deploying functions")
        c.substep("my_function", detail="deploying")
        time.sleep(0.5)
        c.substep("another_function", detail="deploying")
        time.sleep(0.5)
        c.substep("Status: Building... 45%")
        time.sleep(0.3)
        c.substep("Status: Building... 78%")
        time.sleep(0.3)
        c.substep("Status: Building... 100%")
        time.sleep(0.2)
        c.success("Functions deployed", duration=1.8)
        
        # Test info message
        c.info("All resources ready", detail="3 functions active")
        
        # Test separator
        c.separator()
        
        # Final summary
        c.final(
            "Deployment complete",
            details=[
                "View logs: targon logs my-app",
                "Dashboard: https://app.targon.io/my-app",
            ]
        )


if __name__ == "__main__":
    main()


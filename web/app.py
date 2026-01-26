#!/usr/bin/env python3
"""A simple web application to serve documentation for the cvxsimulator package."""

import http.server
import os
import socketserver
import sys
from pathlib import Path

import markdown
from jinja2 import Environment, FileSystemLoader

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Paths
templates_dir = project_root / "web" / "templates"
readme_path = project_root / "README.md"
license_path = project_root / "LICENSE"


# Set up Jinja2 environment
env = Environment(loader=FileSystemLoader(templates_dir), autoescape=True)


def read_readme():
    """Read and convert README.md to HTML."""
    try:
        with open(readme_path, encoding="utf-8") as f:
            readme_content = f.read()

        # Convert Markdown to HTML
        return markdown.markdown(
            readme_content,
            extensions=[
                "markdown.extensions.fenced_code",
                "markdown.extensions.tables",
                "markdown.extensions.codehilite",
            ],
        )
    except Exception as e:
        print(f"Error reading README.md: {e}")
        return "<p>Error loading README content.</p>"


def read_license():
    """Read LICENSE file."""
    try:
        with open(license_path, encoding="utf-8") as f:
            license_content = f.read()
    except Exception as e:
        print(f"Error reading LICENSE: {e}")
        return "<p>Error loading LICENSE content.</p>"
    else:
        return license_content


# Functions for running tests and coverage have been removed
# Test results are now displayed from the CI/CD pipeline (book.yml)


class WebHandler(http.server.SimpleHTTPRequestHandler):
    """Custom request handler for our web server."""

    def __init__(self, *args, **kwargs):
        """Initialize the handler and set URL constants and cached content."""
        # URLs for the template
        self.docs_url = "https://www.cvxgrp.org/simulator/pdoc/"
        self.tests_url = "https://www.cvxgrp.org/simulator/tests/html-report/report.html?sort=result"
        self.coverage_url = "https://www.cvxgrp.org/simulator/tests/html-coverage/index.html"
        self.license_url = "/license"  # Local route for license
        self.github_url = "https://github.com/cvxgrp/simulator"
        self.pypi_url = "https://badge.fury.io/py/cvxsimulator"
        self.downloads_url = "https://pepy.tech/project/cvxsimulator"
        self.coverage_badge_url = "https://coveralls.io/github/cvxgrp/simulator?branch=main"
        self.renovate_url = "https://github.com/renovatebot/renovate"
        self.codespaces_url = "https://codespaces.new/cvxgrp/simulator"
        self.cvxgrp_url = "https://www.cvxgrp.org"

        # Read README and LICENSE content
        self.readme_html = read_readme()
        self.license_text = read_license()

        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/" or self.path == "/index.html":
            self.serve_index()
        elif self.path == "/license" or self.path == "/license.html":
            self.serve_license()
        else:
            # Serve static files
            super().do_GET()

    def serve_index(self):
        """Serve the index page."""
        template = env.get_template("base.html.j2")
        content = template.render(
            readme_html=self.readme_html,
            docs_url=self.docs_url,
            tests_url=self.tests_url,
            coverage_url=self.coverage_url,
            license_url=self.license_url,
            github_url=self.github_url,
            pypi_url=self.pypi_url,
            downloads_url=self.downloads_url,
            coverage_badge_url=self.coverage_badge_url,
            renovate_url=self.renovate_url,
            codespaces_url=self.codespaces_url,
            cvxgrp_url=self.cvxgrp_url,
        )
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(content.encode())

    def serve_license(self):
        """Serve the license content."""
        # Create a simple HTML page with the license text in a pre tag
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>cvxsimulator - License</title>
    <style>
        body {{ font-family: monospace; padding: 20px; }}
        pre {{ white-space: pre-wrap; }}
    </style>
</head>
<body>
    <pre>{self.license_text}</pre>
</body>
</html>"""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(html.encode())


def main():
    """Run the web server."""
    port = 8000
    handler = WebHandler

    # Set the directory to serve static files from
    os.chdir(project_root / "web")

    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"Serving at http://localhost:{port}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()

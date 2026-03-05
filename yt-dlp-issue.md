This is a follow-up to #14404, which announced that yt-dlp will soon require an external JavaScript runtime (e.g. Deno) in order to fully support downloading from YouTube.

With the release of yt-dlp version 2025.11.12, external JavaScript runtime support has arrived.
All users who intend to use yt-dlp with YouTube are strongly encouraged to install one of the supported JS runtimes.
The following JavaScript runtimes are currently supported (in order of recommendation, from strongest to weakest):

Deno
recommended for most users
https://deno.com/
https://github.com/denoland/deno
note: if downloading from Deno's GitHub releases, get deno not denort
minimum Deno version supported by yt-dlp: 2.0.0
the latest version of Deno is strongly recommended
Node
https://nodejs.org/
minimum Node version supported by yt-dlp: 20.0.0
if using Node, the latest version (25+) is strongly recommended for security reasons
QuickJS
https://bellard.org/quickjs/
minimum QuickJS version supported by yt-dlp: 2023-12-9
if using QuickJS, version 2025-4-26 or later is strongly recommended for performance reasons
QuickJS-ng
https://quickjs-ng.github.io/quickjs/
all versions are supported by yt-dlp; however, upstream QuickJS is recommended instead for performance reasons
Bun
https://bun.com/
minimum Bun version supported by yt-dlp: 1.0.31
if using Bun, the latest version is strongly recommended
Note that only deno is enabled by default; all others are disabled by default for security reasons. See the EJS wiki page for more details.

In addition to the JavaScript runtime, yt-dlp also requires the yt-dlp-ejs component in order to operate the JS runtime.

NOTE: This component is already included in all of the official yt-dlp executables.
Similarly, if you've installed & upgraded the yt-dlp Python package with the default extra (yt-dlp[default]), then you already have the yt-dlp-ejs component.

If you've installed yt-dlp another way, then please refer to section 2 of the EJS wiki page for more details.

Support for YouTube without a JavaScript runtime is now considered "deprecated." It does still work somewhat; however, format availability will be limited, and severely so in some cases (e.g. for logged-in users). Format availability without a JS runtime is expected to worsen as time goes on, and this will not be considered a "bug" but rather an inevitability for which there is no solution. It's also expected that, eventually, support for YouTube will not be possible at all without a JS runtime.

If you have questions, please refer to the EJS wiki page, the previous announcement's FAQ, and the README before commenting or opening a new issue:

https://github.com/yt-dlp/yt-dlp/wiki/EJS
[Announcement] Upcoming new requirements for YouTube downloads #14404
https://github.com/yt-dlp/yt-dlp#dependencies
https://github.com/yt-dlp/yt-dlp#general-options
https://github.com/yt-dlp/yt-dlp#youtube-ejs
https://github.com/yt-dlp/yt-dlp/wiki/EJS#plugins
Notes to package maintainers
If you are maintaining a downstream package of yt-dlp, we offer the following guidance:

The yt-dlp repository, source tarball, PyPI source distribution and built distribution (wheel) are still licensed under The Unlicense (public domain); however, when the yt-dlp-ejs package is built, it bundles code licensed under ISC and MIT. This is the primary reason why yt-dlp-ejs was split off into a separate repository and PyPI package

If yt-dlp is packaged as a Python package in your repository, yt-dlp-ejs would ideally be packaged separately

yt-dlp-ejs is technically an optional Python dependency of yt-dlp, but YouTube support is deprecated without it

Each version of yt-dlp will be pinned to a specific version of yt-dlp-ejs and yt-dlp will reject any other yt-dlp-ejs version. Refer to yt-dlp's pyproject.toml for the pinned version

If your repository packages yt-dlp as the zipimport binary instead of as a Python package, you can use make yt-dlp-extra to build the zip executable with yt-dlp-ejs included. (The Makefile will look for the yt-dlp-ejs wheel in the build subdirectory, or the extracted built distribution in the yt_dlp_ejs subdirectory)

deno, nodejs, quickjs and/or bun should be optional dependencies of yt-dlp. But again, YouTube support is deprecated without one of them

While yt-dlp-ejs and the external JavaScript runtimes are currently only used with YouTube, yt-dlp's usage of these may be expanded in the future (and necessarily so)

If this guidance is insufficient, or if you are a developer integrating yt-dlp into your software and you have further questions, please open a new GitHub issue.
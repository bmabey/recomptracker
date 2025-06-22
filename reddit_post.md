**Built a tool to operationalize Attia's ALMI percentile approach - feedback welcome!**

Like many of you, I've been following Attia's approach to body composition and his emphasis on ALMI percentiles over BMI for longevity planning. After getting my DEXA scans, I got tired of manually calculating where I stood relative to the 75th-90th percentile targets he recommends.

So I built **RecompTracker** - a web app that takes your DEXA data and shows exactly where you are on the percentile curves, tracks your progress over time, and calculates what changes you need to hit your goals (weight, lean mass, fat mass). It uses the same LMS reference data from research studies that the medical field uses.

**[Try it here](https://recomptracker.streamlit.app)**

The app handles goal planning with intelligent suggestions based on your training level and progression rate. You can see exactly what it takes to hit 75th, 90th, or whatever percentile you're targeting.

This was also an experiment for me with Claude Code - as a software engineer, I guided the direction but honestly let Claude write almost all the code without looking at it closely. Pretty wild how well it worked for "vibe coding" a complete app.

**[Code is open source](https://github.com/bmabey/recomptracker)**

Future plans include adding bulk/cut cycle planning and more sophisticated goal progression. Hope this helps others on their body comp journey!

Any feedback or feature requests welcome. Curious if others find this useful for tracking their ALMI goals.

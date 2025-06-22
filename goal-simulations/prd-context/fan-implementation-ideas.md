Below are four public examples you can mine for patterns or code when adding a **Monte-Carlo “fan” forecast** to RecompTracker.  Each link either shows the fan-style visual itself or contains code you can copy:

| Where to look                                                                                                                      | What you’ll see / get                                                                                                                                          | Why it’s useful                                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Shedload-of-Code blog – “How to build and visualise a Monte Carlo simulation with Python and Plotly”** ([shedloadofcode.com][1]) | • Step-by-step retirement-fund simulation<br>• Plotly figure with **\~1 000 runs** and a shaded percentile “fan”<br>• Full source on GitHub (link in article)  | Shows how to:<br>`python<br>for run in range(1000):<br>    fig.add_scatter(x=t, y=path, line=dict(color='rgba(0,0,255,0.05)'))`<br>…then add `fill='tonexty'` traces for 50 %/80 % bands. |
| **Streamlit blog – “Monte Carlo simulations with Streamlit”** ([blog.streamlit.io][2])                                             | • Live Streamlit app predicting stock prices with fan chart<br>• Code splits heavy sims into a `@st.cache_data` helper and renders with `st.plotly_chart(fig)` | Demonstrates Streamlit-native caching + Plotly fan rendering inside a multipage app.                                                                                                      |
| **GitHub repo – `WintonCentre/python-fuzzy-plots`** (“FanPlotly” class) ([github.com][3])                                          | • Plug-and-play `FanPlotly` wrapper that takes median + CI arrays and produces a multi-band fan chart                                                          | Fastest path if you want to pass 5th/95th, 25th/75th percentiles from your Monte-Carlo and let the helper build all shaded layers.                                                        |
| **Plotly Community thread – “Monte Carlo plot (1000+ lines)”** ([community.plotly.com][4])                                         | • Minimal example of pushing thousands of traces without performance collapse                                                                                  | Helpful for performance tweaks (e.g., use `go.Scattergl` for WebGL rendering when >5 000 traces).                                                                                         |

### How this fits RecompTracker

1. **Simulation engine** – reuse your rates & P-ratio distributions, run `n=2 000` trajectories.
2. **Fan chart** – plot:

   * Light alpha lines (or omit for speed)
   * Shaded 10–90 % and 25–75 % bands (examples above)
   * Bold median path (pick the trace closest to the median at each time-step).
3. **User controls** – mirror the Streamlit stock app: sliders for confidence band, number of runs, and a checkbox “show individual paths”.
4. **Performance** – if fans feel sluggish, copy the Scattergl trick from the Plotly community example.

These four references give you concrete code and UI patterns you can adapt directly into your Streamlit + Plotly front end.

[1]: https://www.shedloadofcode.com/blog/how-to-build-and-visualise-a-monte-carlo-simulation-with-python-and-plotly/ "How to build and visualise a Monte Carlo simulation with Python and Plotly | Shedload Of Code"
[2]: https://blog.streamlit.io/monte-carlo-simulations-with-streamlit/ "Monte Carlo simulations with Streamlit"
[3]: https://github.com/WintonCentre/python-fuzzy-plots?utm_source=chatgpt.com "WintonCentre/python-fuzzy-plots - GitHub"
[4]: https://community.plotly.com/t/monte-carlo-plot/16225?utm_source=chatgpt.com "Monte Carlo Plot - Plotly Python - Plotly Community Forum"


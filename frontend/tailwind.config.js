/** Palette: night-drive teal with road-sign signal colours. */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        night: "#0d2b30",
        panel: "#14393f",
        raised: "#1b464d",
        line: "#2b5a62",
        paper: "#e9f1ef",
        muted: "#93b3b0",
        ok: "#5fd3a8",
        caution: "#f4b73f",
        danger: "#f0585c",
      },
      fontFamily: {
        sans: ["Barlow", "system-ui", "sans-serif"],
        display: ["Barlow Condensed", "Barlow", "system-ui", "sans-serif"],
      },
    },
  },
  plugins: [],
};

/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      boxShadow: {
        glow: "0 0 22px rgba(34,197,94,0.22), 0 0 34px rgba(245,158,11,0.14)",
        glow2: "0 0 18px rgba(20,184,166,0.18)",
      },
      colors: { ink: "#070A10" },
    },
  },
  plugins: [],
};

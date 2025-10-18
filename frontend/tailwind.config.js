/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'brand-dark-blue': '#0f172a',
        'brand-blue': '#3b82f6',
        'brand-light-blue': '#e0f2fe',
        'brand-gray': '#64748b',
        'status-deny-bg': '#fee2e2',
        'status-deny-text': '#dc2626',
        'status-review-bg': '#fef3c7',
        'status-review-text': '#d97706',
        'status-approve-bg': '#dcfce7',
        'status-approve-text': '#16a34a',
      },
      fontFamily: {
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
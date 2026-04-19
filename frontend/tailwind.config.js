/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        vg: {
          bg:      '#060a10',
          surface: '#0d1117',
          card:    '#0c121e',
          border:  'rgba(56,189,248,0.15)',
          accent:  '#38bdf8',
          muted:   '#475569',
        },
        threat: {
          hostile:  '#ef4444',
          suspect:  '#f59e0b',
          friend:   '#22c55e',
          neutral:  '#94a3b8',
          civilian: '#38bdf8',
        },
      },
      fontFamily: {
        orbitron: ['Orbitron', 'monospace'],
        space:    ['Space Grotesk', 'sans-serif'],
      },
      keyframes: {
        'slide-in': {
          from: { opacity: '0', transform: 'translateX(-12px)' },
          to:   { opacity: '1', transform: 'translateX(0)' },
        },
        'fade-up': {
          from: { opacity: '0', transform: 'translateY(8px)' },
          to:   { opacity: '1', transform: 'translateY(0)' },
        },
        'pulse-dot': {
          '0%,100%': { opacity: '1', transform: 'scale(1)' },
          '50%':     { opacity: '0.35', transform: 'scale(1.5)' },
        },
        'scan': {
          '0%,100%': { opacity: '0.3', transform: 'scaleX(0.4)' },
          '50%':     { opacity: '1',   transform: 'scaleX(1)' },
        },
      },
      animation: {
        'slide-in':  'slide-in 0.35s ease-out forwards',
        'fade-up':   'fade-up 0.4s ease-out forwards',
        'pulse-dot': 'pulse-dot 2.2s ease-in-out infinite',
        'scan':      'scan 3s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}

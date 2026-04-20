import React, { useEffect, useRef, useState, useCallback } from 'react'
import { useParams, Link } from 'react-router-dom'
import * as THREE from 'three'
import type { SimTrack } from '../sim-types'
import type { Track } from '../types'
import { CLASS_STYLES } from '../types'
import Trail3D from '../components/Trail3D'
import AnomalyAlerts from '../components/AnomalyAlerts'

const BASE = import.meta.env.VITE_API_URL ?? ''

function getWsUrl(sid: string): string {
  if (BASE) return BASE.replace(/^https:\/\//, 'wss://').replace(/^http:\/\//, 'ws://') + `/sim/${sid}/ws`
  return `ws://${window.location.host}/sim/${sid}/ws`
}

// ── Coordinate helpers ────────────────────────────────────────────────────────
function toSvg(pos: { x: number; y: number }) {
  return { sx: 500 + pos.x * 5, sy: 300 + pos.y * 5 }
}
// SVG coords + altitude → Three.js world
function svgTo3D(svgX: number, svgY: number, altFt: number): THREE.Vector3 {
  return new THREE.Vector3(svgX - 500, altFt / 260, svgY - 300)
}

// ── Commercial routes (SVG coord space) ──────────────────────────────────────
const COM_ROUTES = [
  { id: 'N600',  x1: 958, y1:  60, x2:  55, y2: 132 },
  { id: 'UL613', x1:  55, y1: 468, x2: 958, y2: 528 },
  { id: 'B57',   x1: 940, y1: 236, x2:  80, y2: 196 },
]
const COM_AC = [
  { id:'THY401', route:'N600',  phase:0.08, spd:0.00055, lbl:'THY401', fl:'FL370' },
  { id:'BAW156', route:'N600',  phase:0.43, spd:0.00070, lbl:'BAW156', fl:'FL360' },
  { id:'DLH892', route:'N600',  phase:0.77, spd:0.00065, lbl:'DLH892', fl:'FL350' },
  { id:'UAE215', route:'UL613', phase:0.18, spd:0.00060, lbl:'UAE215', fl:'FL390' },
  { id:'TKY789', route:'UL613', phase:0.63, spd:0.00075, lbl:'TKY789', fl:'FL380' },
  { id:'AAL102', route:'B57',   phase:0.22, spd:0.00068, lbl:'AAL102', fl:'FL340' },
  { id:'UAL567', route:'B57',   phase:0.67, spd:0.00085, lbl:'UAL567', fl:'FL350' },
]
function comPos(ac: typeof COM_AC[0], tick: number) {
  const r = COM_ROUTES.find(r => r.id === ac.route)!
  const t = ((ac.phase + tick * ac.spd) % 1 + 1) % 1
  return {
    x: r.x1 + (r.x2 - r.x1) * t,
    y: r.y1 + (r.y2 - r.y1) * t,
    angle: Math.atan2(r.y2 - r.y1, r.x2 - r.x1) * 180 / Math.PI,
  }
}

// ── Animated track position ───────────────────────────────────────────────────
function animPos(track: SimTrack, tick: number) {
  const base = toSvg(track.pos)
  const t = tick * 0.018
  if (track.submitted_at !== 'DEMO') {
    return { sx: base.sx + Math.sin(t * 2.1) * 1.2, sy: base.sy + Math.cos(t * 1.7) * 0.9 }
  }
  switch (track.ai_class) {
    case 'HOSTILE':        return { sx: base.sx + Math.sin(t * 0.22) * 22, sy: base.sy + Math.cos(t * 0.31) * 7 }
    case 'SUSPECT':        return { sx: base.sx + Math.cos(t * 0.65) * 16, sy: base.sy + Math.sin(t * 0.65) * 9 }
    case 'UNKNOWN':        return { sx: base.sx + Math.sin(t * 0.4) * 6,   sy: base.sy + Math.sin(t * 0.18) * 22 }
    case 'NEUTRAL':        return { sx: base.sx + Math.sin(t * 0.13) * 28, sy: base.sy + Math.sin(t * 0.35) * 3 }
    case 'ASSUMED FRIEND': return { sx: base.sx + Math.cos(t * 0.55) * 11, sy: base.sy + Math.sin(t * 0.80) * 7 }
    case 'FRIEND':         return { sx: base.sx + Math.cos(t * 0.45) * 9,  sy: base.sy + Math.sin(t * 0.45) * 6 }
    default:               return base
  }
}

// ── Heading vector (2D view) ──────────────────────────────────────────────────
function hdgVec(deg: number, len = 14) {
  const r = (deg - 90) * Math.PI / 180
  return { dx: Math.cos(r) * len, dy: Math.sin(r) * len }
}

// ── NATO class → Three.js hex color ──────────────────────────────────────────
const CLASS_COLOR_HEX: Record<string, number> = {
  'HOSTILE':        0xff2222,
  'SUSPECT':        0xff8800,
  'UNKNOWN':        0xaaaaaa,
  'NEUTRAL':        0xddcc00,
  'ASSUMED FRIEND': 0x4499ff,
  'FRIEND':         0x22ff88,
}

// ── Demo tracks ───────────────────────────────────────────────────────────────
const DEMO_TRACKS: SimTrack[] = [
  {
    track_id:'SIM-H001', submitted_at:'DEMO', ai_class:'HOSTILE', ai_conf:0.91,
    ai_probs:{ HOSTILE:0.91, SUSPECT:0.05, UNKNOWN:0.02, NEUTRAL:0.01, 'ASSUMED FRIEND':0.005, FRIEND:0.005 },
    pos:{x:-62,y:-8},
    sensor_votes:{
      radar:{label:'Radar',icon:'📡',vote:'HOSTILE',conf:0.88,reading:'RCS 0.02 m² · 920 kts'},
      esm:  {label:'ESM',  icon:'📻',vote:'HOSTILE',conf:0.93,reading:'HOSTILE JAMMING'},
      irst: {label:'IRST', icon:'🌡️',vote:'SUSPECT',conf:0.76,reading:'High thermal'},
      iff:  {label:'IFF',  icon:'🔑',vote:'SUSPECT',conf:0.84,reading:'NO RESPONSE'},
    },
    fusion:{best:'HOSTILE',probs:{HOSTILE:0.91,SUSPECT:0.05,UNKNOWN:0.02,NEUTRAL:0.01,'ASSUMED FRIEND':0.005,FRIEND:0.005},weights:{}},
    xai:[],
    altitude_ft:500, speed_kts:920, rcs_m2:0.02, heading:90,
    esm_signature:'HOSTILE_JAMMING', iff_mode:'NO_RESPONSE', flight_profile:'LOW_ALTITUDE_FLYING', weather:'Clear', thermal_signature:'High',
  },
  {
    track_id:'SIM-S002', submitted_at:'DEMO', ai_class:'SUSPECT', ai_conf:0.78,
    ai_probs:{HOSTILE:0.12,SUSPECT:0.78,UNKNOWN:0.06,NEUTRAL:0.02,'ASSUMED FRIEND':0.01,FRIEND:0.01},
    pos:{x:-52,y:-28},
    sensor_votes:{
      radar:{label:'Radar',icon:'📡',vote:'SUSPECT',conf:0.72,reading:'RCS 0.8 m² · 580 kts'},
      esm:  {label:'ESM',  icon:'📻',vote:'SUSPECT',conf:0.81,reading:'NOISE JAMMING'},
      irst: {label:'IRST', icon:'🌡️',vote:'SUSPECT',conf:0.68,reading:'Medium thermal'},
      iff:  {label:'IFF',  icon:'🔑',vote:'SUSPECT',conf:0.84,reading:'NO RESPONSE'},
    },
    fusion:{best:'SUSPECT',probs:{HOSTILE:0.12,SUSPECT:0.78,UNKNOWN:0.06,NEUTRAL:0.02,'ASSUMED FRIEND':0.01,FRIEND:0.01},weights:{}},
    xai:[],
    altitude_ft:8000, speed_kts:580, rcs_m2:0.8, heading:115,
    esm_signature:'NOISE_JAMMING', iff_mode:'NO_RESPONSE', flight_profile:'AGGRESSIVE_MANEUVERS', weather:'Cloudy', thermal_signature:'Medium',
  },
  {
    track_id:'SIM-U003', submitted_at:'DEMO', ai_class:'UNKNOWN', ai_conf:0.62,
    ai_probs:{HOSTILE:0.08,SUSPECT:0.14,UNKNOWN:0.62,NEUTRAL:0.10,'ASSUMED FRIEND':0.04,FRIEND:0.02},
    pos:{x:8,y:-57},
    sensor_votes:{
      radar:{label:'Radar',icon:'📡',vote:'UNKNOWN',conf:0.55,reading:'RCS 1.5 m² · 460 kts'},
      esm:  {label:'ESM',  icon:'📻',vote:'UNKNOWN',conf:0.52,reading:'UNKNOWN EMISSION'},
      irst: {label:'IRST', icon:'🌡️',vote:'NEUTRAL',conf:0.48,reading:'Low thermal'},
      iff:  {label:'IFF',  icon:'🔑',vote:'UNKNOWN',conf:0.55,reading:'DEGRADED'},
    },
    fusion:{best:'UNKNOWN',probs:{HOSTILE:0.08,SUSPECT:0.14,UNKNOWN:0.62,NEUTRAL:0.10,'ASSUMED FRIEND':0.04,FRIEND:0.02},weights:{}},
    xai:[],
    altitude_ft:28000, speed_kts:460, rcs_m2:1.5, heading:180,
    esm_signature:'UNKNOWN_EMISSION', iff_mode:'DEGRADED', flight_profile:'STABLE_CRUISE', weather:'Cloudy', thermal_signature:'Low',
  },
  {
    track_id:'SIM-N004', submitted_at:'DEMO', ai_class:'NEUTRAL', ai_conf:0.74,
    ai_probs:{HOSTILE:0.02,SUSPECT:0.05,UNKNOWN:0.12,NEUTRAL:0.74,'ASSUMED FRIEND':0.05,FRIEND:0.02},
    pos:{x:-5,y:42},
    sensor_votes:{
      radar:{label:'Radar',icon:'📡',vote:'NEUTRAL',conf:0.70,reading:'RCS 15 m² · 440 kts'},
      esm:  {label:'ESM',  icon:'📻',vote:'NEUTRAL',conf:0.72,reading:'CLEAN'},
      irst: {label:'IRST', icon:'🌡️',vote:'NEUTRAL',conf:0.65,reading:'Low thermal'},
      iff:  {label:'IFF',  icon:'🔑',vote:'ASSUMED FRIEND',conf:0.88,reading:'IFF MODE 3C'},
    },
    fusion:{best:'NEUTRAL',probs:{HOSTILE:0.02,SUSPECT:0.05,UNKNOWN:0.12,NEUTRAL:0.74,'ASSUMED FRIEND':0.05,FRIEND:0.02},weights:{}},
    xai:[],
    altitude_ft:36000, speed_kts:440, rcs_m2:15, heading:270,
    esm_signature:'CLEAN', iff_mode:'IFF_MODE_3C', flight_profile:'STABLE_CRUISE', weather:'Clear', thermal_signature:'Low',
  },
  {
    track_id:'SIM-A005', submitted_at:'DEMO', ai_class:'ASSUMED FRIEND', ai_conf:0.83,
    ai_probs:{HOSTILE:0.01,SUSPECT:0.02,UNKNOWN:0.05,NEUTRAL:0.09,'ASSUMED FRIEND':0.83,FRIEND:0.00},
    pos:{x:56,y:-20},
    sensor_votes:{
      radar:{label:'Radar',icon:'📡',vote:'ASSUMED FRIEND',conf:0.79,reading:'RCS 3 m² · 400 kts'},
      esm:  {label:'ESM',  icon:'📻',vote:'NEUTRAL',conf:0.72,reading:'CLEAN'},
      irst: {label:'IRST', icon:'🌡️',vote:'NEUTRAL',conf:0.65,reading:'Medium thermal'},
      iff:  {label:'IFF',  icon:'🔑',vote:'ASSUMED FRIEND',conf:0.88,reading:'IFF MODE 3C'},
    },
    fusion:{best:'ASSUMED FRIEND',probs:{HOSTILE:0.01,SUSPECT:0.02,UNKNOWN:0.05,NEUTRAL:0.09,'ASSUMED FRIEND':0.83,FRIEND:0.00},weights:{}},
    xai:[],
    altitude_ft:22000, speed_kts:400, rcs_m2:3, heading:270,
    esm_signature:'CLEAN', iff_mode:'IFF_MODE_3C', flight_profile:'STABLE_CRUISE', weather:'Clear', thermal_signature:'Medium',
  },
  {
    track_id:'SIM-F006', submitted_at:'DEMO', ai_class:'FRIEND', ai_conf:0.96,
    ai_probs:{HOSTILE:0.005,SUSPECT:0.005,UNKNOWN:0.01,NEUTRAL:0.02,'ASSUMED FRIEND':0.00,FRIEND:0.96},
    pos:{x:64,y:18},
    sensor_votes:{
      radar:{label:'Radar',icon:'📡',vote:'FRIEND',conf:0.90,reading:'RCS 2 m² · 520 kts'},
      esm:  {label:'ESM',  icon:'📻',vote:'NEUTRAL',conf:0.72,reading:'CLEAN'},
      irst: {label:'IRST', icon:'🌡️',vote:'NEUTRAL',conf:0.65,reading:'Medium thermal'},
      iff:  {label:'IFF',  icon:'🔑',vote:'FRIEND',conf:0.98,reading:'IFF MODE 5'},
    },
    fusion:{best:'FRIEND',probs:{HOSTILE:0.005,SUSPECT:0.005,UNKNOWN:0.01,NEUTRAL:0.02,'ASSUMED FRIEND':0.00,FRIEND:0.96},weights:{}},
    xai:[],
    altitude_ft:18000, speed_kts:520, rcs_m2:2, heading:90,
    esm_signature:'CLEAN', iff_mode:'IFF_MODE_5', flight_profile:'CLIMBING', weather:'Clear', thermal_signature:'Medium',
  },
]

const SPOKES = [0,45,90,135,180,225,270,315].map(deg => {
  const r = (deg-90)*Math.PI/180
  return { x2: 500+330*Math.cos(r), y2: 300+330*Math.sin(r) }
})

// ── Track symbol (2D) ─────────────────────────────────────────────────────────
function TrackSymbol({ cls }: { cls: string }) {
  const color = CLASS_STYLES[cls]?.color ?? '#888'
  const S = 9
  switch (cls) {
    case 'HOSTILE':        return <polygon points={`0,${S} ${S},${-S} ${-S},${-S}`} fill={color}/>
    case 'SUSPECT':        return <polygon points={`0,${-S} ${S},0 0,${S} ${-S},0`} fill="none" stroke={color} strokeWidth={2.5}/>
    case 'UNKNOWN':        return <circle r={S} fill="none" stroke={color} strokeWidth={2.5}/>
    case 'NEUTRAL':        return <rect x={-S} y={-S} width={S*2} height={S*2} fill="none" stroke={color} strokeWidth={2}/>
    case 'ASSUMED FRIEND': return <rect x={-S} y={-S} width={S*2} height={S*2} fill={color} opacity={0.7}/>
    case 'FRIEND':         return <polygon points={`0,${-S} ${S},${S} ${-S},${S}`} fill={color}/>
    default:               return <circle r={S} fill={color} opacity={0.5}/>
  }
}

// ── 3D Tactical Map (Three.js) ────────────────────────────────────────────────
interface TrackObj3D {
  cone: THREE.Mesh
  pole: THREE.Line
  hdgLine: THREE.Line
  trackData: SimTrack
}

function addGroundPlane(scene: THREE.Scene, w: number, d: number, color: number, px: number, pz: number) {
  const geo = new THREE.PlaneGeometry(w, d)
  geo.rotateX(-Math.PI / 2)
  const mat = new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(px, -0.3, pz)
  scene.add(mesh)
}

function TacticalMap3D({ tracks, tick, onInspect }: {
  tracks: SimTrack[]
  tick: number
  onInspect: (t: SimTrack) => void
}) {
  const mountRef   = useRef<HTMLDivElement>(null)
  const rendRef    = useRef<THREE.WebGLRenderer | null>(null)
  const sceneRef   = useRef<THREE.Scene | null>(null)
  const camRef     = useRef<THREE.PerspectiveCamera | null>(null)
  const trackObjs  = useRef<Map<string, TrackObj3D>>(new Map())
  const comMeshes  = useRef<Map<string, THREE.Mesh>>(new Map())
  const rafRef     = useRef<number>(0)
  const camAngle   = useRef(0.28)          // horizontal orbit
  const camElev    = useRef(0.52)          // vertical elevation
  const camDist    = useRef(800)
  const isDrag     = useRef(false)
  const dragStart  = useRef({ x: 0, y: 0 })

  // Update camera from refs — safe as plain function (only reads refs)
  function updateCam() {
    const cam = camRef.current
    if (!cam) return
    const cosE = Math.cos(camElev.current)
    cam.position.set(
      camDist.current * Math.sin(camAngle.current) * cosE,
      camDist.current * Math.sin(camElev.current),
      camDist.current * Math.cos(camAngle.current) * cosE,
    )
    cam.lookAt(0, 55, 0)
  }

  // ── Scene setup (once) ──────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current
    if (!mount) return
    const W = mount.clientWidth  || 900
    const H = mount.clientHeight || 600

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.setSize(W, H)
    renderer.setClearColor(0x060d19)
    mount.appendChild(renderer.domElement)
    rendRef.current = renderer

    // Camera
    const camera = new THREE.PerspectiveCamera(50, W / H, 1, 5000)
    camRef.current = camera
    updateCam()

    // Scene
    const scene = new THREE.Scene()
    scene.fog = new THREE.FogExp2(0x060d19, 0.00038)
    sceneRef.current = scene

    // Lighting
    scene.add(new THREE.AmbientLight(0x1a2840, 3))
    const sun = new THREE.DirectionalLight(0x4466aa, 1.4)
    sun.position.set(300, 600, 200)
    scene.add(sun)

    // Ground: enemy (left), sea (centre), friendly (right)
    addGroundPlane(scene, 500, 800, 0x160505,  -430, 0)  // enemy
    addGroundPlane(scene, 460, 800, 0x040c1a,     0, 0)  // sea
    addGroundPlane(scene, 500, 800, 0x050516,   430, 0)  // friendly

    // Grid
    const grid = new THREE.GridHelper(900, 18, 0x061424, 0x061424)
    grid.position.y = 0.5
    scene.add(grid)

    // Boundary lines (threat / friendly separators)
    ;[-200, 200].forEach((x, i) => {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, 2, -380),
        new THREE.Vector3(x, 2,  380),
      ])
      scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: i === 0 ? 0x6a1414 : 0x141470,
        transparent: true, opacity: 0.55,
      })))
    })

    // Range rings on ground
    ;[100, 200, 300].forEach(r => {
      const pts: THREE.Vector3[] = []
      for (let i = 0; i <= 72; i++) {
        const a = (i / 72) * Math.PI * 2
        pts.push(new THREE.Vector3(Math.cos(a) * r, 1, Math.sin(a) * r))
      }
      const geo = new THREE.BufferGeometry().setFromPoints(pts)
      scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: 0x083858, transparent: true, opacity: 0.5,
      })))
    })

    // Azimuth spokes
    for (let deg = 0; deg < 360; deg += 45) {
      const rad = deg * Math.PI / 180
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 1, 0),
        new THREE.Vector3(Math.sin(rad) * 340, 1, -Math.cos(rad) * 340),
      ])
      scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: 0x061828, transparent: true, opacity: 0.3,
      })))
    }

    // Vertical altitude axis
    {
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, 0),
        new THREE.Vector3(0, 170, 0),
      ])
      scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: 0x0a3a4a, transparent: true, opacity: 0.4,
      })))
    }

    // Altitude tick marks on axis
    ;[10000, 20000, 30000, 40000].forEach(ft => {
      const y = ft / 260
      const geo = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-8, y, 0), new THREE.Vector3(8, y, 0),
      ])
      scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: 0x0a3a4a, transparent: true, opacity: 0.45,
      })))
    })

    // Commercial routes at cruise altitude
    const cruiseY = 36000 / 260
    COM_ROUTES.forEach(route => {
      const p1 = svgTo3D(route.x1, route.y1, 36000)
      const p2 = svgTo3D(route.x2, route.y2, 36000)
      const geo = new THREE.BufferGeometry().setFromPoints([p1, p2])
      scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: 0x2244bb, transparent: true, opacity: 0.3,
      })))
      // Vertical curtain lines at endpoints
      ;[p1, p2].forEach(p => {
        const g2 = new THREE.BufferGeometry().setFromPoints([
          new THREE.Vector3(p.x, 1, p.z),
          new THREE.Vector3(p.x, cruiseY, p.z),
        ])
        scene.add(new THREE.Line(g2, new THREE.LineBasicMaterial({
          color: 0x1133aa, transparent: true, opacity: 0.12,
        })))
      })
    })

    // Own ship (octahedron + point light)
    const shipGeo = new THREE.OctahedronGeometry(16, 0)
    const shipMat = new THREE.MeshPhongMaterial({
      color: 0x00ccdd, emissive: 0x003344, shininess: 90,
      transparent: true, opacity: 0.88,
    })
    const ship = new THREE.Mesh(shipGeo, shipMat)
    ship.position.set(0, 8, 0)
    scene.add(ship)
    const shipLight = new THREE.PointLight(0x00ffff, 4, 240)
    shipLight.position.set(0, 12, 0)
    scene.add(shipLight)

    // Render loop
    let alive = true
    function loop() {
      if (!alive) return
      rafRef.current = requestAnimationFrame(loop)
      renderer.render(scene, camera)
    }
    loop()

    // Resize handler
    function onResize() {
      const W = mount!.clientWidth, H = mount!.clientHeight
      camera.aspect = W / H
      camera.updateProjectionMatrix()
      renderer.setSize(W, H)
    }
    window.addEventListener('resize', onResize)

    return () => {
      alive = false
      cancelAnimationFrame(rafRef.current)
      window.removeEventListener('resize', onResize)
      if (mount.contains(renderer.domElement)) mount.removeChild(renderer.domElement)
      renderer.dispose()
      rendRef.current = null
      sceneRef.current = null
      camRef.current = null
      trackObjs.current.clear()
      comMeshes.current.clear()
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Update track meshes ─────────────────────────────────────────────────────
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return

    const liveIds = new Set(tracks.map(t => t.track_id))

    // Remove stale
    trackObjs.current.forEach((obj, id) => {
      if (!liveIds.has(id)) {
        scene.remove(obj.cone, obj.pole, obj.hdgLine)
        obj.cone.geometry.dispose();
        (obj.cone.material as THREE.Material).dispose()
        obj.pole.geometry.dispose()
        obj.hdgLine.geometry.dispose()
        trackObjs.current.delete(id)
      }
    })

    tracks.forEach(track => {
      const svgP = animPos(track, tick)
      const pos  = svgTo3D(svgP.sx, svgP.sy, track.altitude_ft)
      const hex  = CLASS_COLOR_HEX[track.ai_class] ?? 0x888888

      if (!trackObjs.current.has(track.track_id)) {
        // Cone — apex up, coloured by class
        const coneGeo = new THREE.ConeGeometry(8, 22, 8)
        const coneMat = new THREE.MeshPhongMaterial({
          color: hex, emissive: hex, emissiveIntensity: 0.35,
          transparent: true, opacity: 0.90, shininess: 70,
        })
        const cone = new THREE.Mesh(coneGeo, coneMat)
        cone.userData.trackId = track.track_id
        scene.add(cone)

        // Altitude pole
        const polePts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 1, 0)]
        const poleGeo = new THREE.BufferGeometry().setFromPoints(polePts)
        const pole    = new THREE.Line(poleGeo, new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.28 }))
        scene.add(pole)

        // Heading line (horizontal at altitude)
        const hdgPts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, 0, 1)]
        const hdgGeo = new THREE.BufferGeometry().setFromPoints(hdgPts)
        const hdgLine = new THREE.Line(hdgGeo, new THREE.LineBasicMaterial({ color: hex, transparent: true, opacity: 0.55 }))
        scene.add(hdgLine)

        trackObjs.current.set(track.track_id, { cone, pole, hdgLine, trackData: track })
      }

      const obj = trackObjs.current.get(track.track_id)!
      obj.trackData = track

      // Cone position
      obj.cone.position.copy(pos)
      // Slow spin for hostile/suspect
      if (track.ai_class === 'HOSTILE' || track.ai_class === 'SUSPECT') {
        obj.cone.rotation.y = tick * 0.04
      }

      // Altitude pole
      const pp = obj.pole.geometry.attributes.position as THREE.BufferAttribute
      pp.setXYZ(0, pos.x, 1,     pos.z)
      pp.setXYZ(1, pos.x, pos.y, pos.z)
      pp.needsUpdate = true

      // Heading line
      const hdgRad = track.heading * Math.PI / 180
      const hdgLen = 32
      const hp = obj.hdgLine.geometry.attributes.position as THREE.BufferAttribute
      hp.setXYZ(0, pos.x,                       pos.y + 3, pos.z)
      hp.setXYZ(1, pos.x + Math.sin(hdgRad)*hdgLen, pos.y + 3, pos.z - Math.cos(hdgRad)*hdgLen)
      hp.needsUpdate = true
    })
  }, [tracks, tick])

  // ── Update commercial aircraft ──────────────────────────────────────────────
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return
    COM_AC.forEach(ac => {
      const cp  = comPos(ac, tick)
      const pos = svgTo3D(cp.x, cp.y, 36000)
      if (!comMeshes.current.has(ac.id)) {
        const geo  = new THREE.ConeGeometry(5, 13, 6)
        const mat  = new THREE.MeshPhongMaterial({ color: 0x7788bb, transparent: true, opacity: 0.55 })
        const mesh = new THREE.Mesh(geo, mat)
        mesh.userData.commercial = true
        scene.add(mesh)
        comMeshes.current.set(ac.id, mesh)
      }
      comMeshes.current.get(ac.id)!.position.copy(pos)
    })
  }, [tick])

  // ── Mouse handlers ──────────────────────────────────────────────────────────
  function onMouseDown(e: React.MouseEvent) {
    isDrag.current  = true
    dragStart.current = { x: e.clientX, y: e.clientY }
  }
  function onMouseMove(e: React.MouseEvent) {
    if (!isDrag.current) return
    const dx = e.clientX - dragStart.current.x
    const dy = e.clientY - dragStart.current.y
    camAngle.current += dx * 0.007
    camElev.current   = Math.max(0.08, Math.min(Math.PI/2 - 0.05, camElev.current - dy * 0.005))
    updateCam()
    dragStart.current = { x: e.clientX, y: e.clientY }
  }
  function onMouseUp(e: React.MouseEvent) {
    const moved = Math.abs(e.clientX - dragStart.current.x) + Math.abs(e.clientY - dragStart.current.y)
    isDrag.current = false
    if (moved < 5) handleClick(e.clientX, e.clientY)
  }
  function handleClick(cx: number, cy: number) {
    const mount = mountRef.current
    const cam   = camRef.current
    if (!mount || !cam) return
    const rect  = mount.getBoundingClientRect()
    const mouse = new THREE.Vector2(
      ((cx - rect.left) / rect.width)  *  2 - 1,
      ((cy - rect.top)  / rect.height) * -2 + 1,
    )
    const ray   = new THREE.Raycaster()
    ray.setFromCamera(mouse, cam)
    const cones = [...trackObjs.current.values()].map(o => o.cone)
    const hits  = ray.intersectObjects(cones)
    if (hits.length > 0) {
      const id  = hits[0].object.userData.trackId as string
      const obj = trackObjs.current.get(id)
      if (obj) onInspect(obj.trackData)
    }
  }
  function onWheel(e: React.WheelEvent) {
    camDist.current = Math.max(200, Math.min(1800, camDist.current + e.deltaY * 0.55))
    updateCam()
  }

  // hex → css rgb string
  function hexCss(hex: number) {
    return `rgb(${(hex>>16)&0xff},${(hex>>8)&0xff},${hex&0xff})`
  }

  return (
    <div
      ref={mountRef}
      className="w-full h-full relative overflow-hidden"
      style={{ cursor: isDrag.current ? 'grabbing' : 'grab' }}
      onMouseDown={onMouseDown}
      onMouseMove={onMouseMove}
      onMouseUp={onMouseUp}
      onMouseLeave={() => { isDrag.current = false }}
      onWheel={onWheel}
    >
      {/* Top-left HUD */}
      <div className="absolute top-3 left-3 pointer-events-none z-10 space-y-1">
        <div className="text-cyan-400/40 text-[10px] font-mono tracking-widest">
          3D AIRSPACE · DRAG ROTATE · SCROLL ZOOM · CLICK TRACK
        </div>
        {/* Altitude ruler */}
        <div className="mt-2 flex flex-col items-start gap-0.5">
          {[{ft:40000,lbl:'FL400'},{ft:30000,lbl:'FL300'},{ft:20000,lbl:'FL200'},{ft:10000,lbl:'FL100'},{ft:0,lbl:'GND'}].map(({ft,lbl})=>(
            <div key={ft} className="flex items-center gap-2">
              <div className="w-5 h-px bg-cyan-400/20"/>
              <span className="text-[9px] font-mono text-slate-600">{lbl}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Bottom-left legend */}
      <div className="absolute bottom-4 left-3 pointer-events-none z-10 space-y-1">
        {Object.entries(CLASS_COLOR_HEX).map(([cls, hex]) => (
          <div key={cls} className="flex items-center gap-2">
            <div className="w-2.5 h-2.5 rounded-sm" style={{ backgroundColor: hexCss(hex) }}/>
            <span className="text-[10px] font-mono" style={{ color: hexCss(hex) }}>{cls}</span>
          </div>
        ))}
      </div>

      {/* Bottom-right: commercial traffic note */}
      <div className="absolute bottom-4 right-3 pointer-events-none z-10 text-right">
        <div className="text-[9px] font-mono text-slate-700">✈ COMMERCIAL FL340-FL390</div>
        <div className="text-[9px] font-mono text-slate-700">◆ OWN SHIP</div>
      </div>
    </div>
  )
}

// ── Sim trail adapter ─────────────────────────────────────────────────────────
// Converts SimTrack → minimal Track shape that Trail3D needs.
// If backend provided hist_* arrays, use them directly.
// Demo tracks fall back to a deterministic synthetic profile.
function simToTrailTrack(t: SimTrack): Track {
  let alts      = t.hist_alts
  let spds      = t.hist_speeds
  let hdgs      = t.hist_headings
  let times     = t.hist_timestamps

  if (!alts || alts.length === 0) {
    // Synthetic fallback (demo tracks)
    const N   = 14
    const seed = t.track_id.split('').reduce((a,c) => a + c.charCodeAt(0), 0)
    let s = (seed * 9301 + 49297) % 233280
    const rng = () => { s = (s * 9301 + 49297) % 233280; return s / 233280 }

    const PARAMS: Record<string, [number, number, number, number]> = {
      LOW_ALTITUDE_FLYING:  [-20000, 110, 10, 400],
      AGGRESSIVE_MANEUVERS: [  4000,  85, 32, 1600],
      CLIMBING:             [-12000,  40,  4, 300],
      STABLE_CRUISE:        [   600,  15,  3, 200],
    }
    const [altT, spdT, hdgJ, altN] = PARAMS[t.flight_profile] ?? PARAMS.STABLE_CRUISE

    alts = []; spds = []; hdgs = []; times = []
    let hdgAcc = t.heading
    for (let i = 0; i < N; i++) {
      const f = i / (N - 1)
      alts.push(Math.max(200, Math.round(t.altitude_ft + altT * (1-f) + (rng()-0.5)*altN)))
      spds.push(Math.max(120, Math.round(t.speed_kts  + spdT * (1-f) + (rng()-0.5)*20)))
      if (i < N-1) { hdgAcc = (hdgAcc + (rng()-0.5)*hdgJ*2) % 360 }
      hdgs.push(Math.round(i === N-1 ? t.heading : (hdgAcc + 360) % 360))
      const ago = (N-1-i)*5
      times.push(ago === 0 ? 'NOW' : `T-${ago}m`)
    }
    alts[N-1] = t.altitude_ft; spds[N-1] = Math.round(t.speed_kts); hdgs[N-1] = t.heading
  }

  return {
    track_id:        t.track_id,
    ai_class:        t.ai_class,
    speed_kts:       t.speed_kts,
    heading:         t.heading,
    hist_alts:       alts,
    hist_speeds:     spds,
    hist_headings:   hdgs,
    hist_timestamps: times,
  } as unknown as Track
}

// ── Inspect modal ─────────────────────────────────────────────────────────────
function InspectModal({ track, onClose }: { track: SimTrack; onClose: () => void }) {
  const s = CLASS_STYLES[track.ai_class]
  const backdropRef = useRef<HTMLDivElement>(null)

  const fusionProbs = track.ai_probs ?? track.fusion?.probs ?? {}
  const sortedProbs = Object.entries(fusionProbs).sort(([,a],[,b]) => b-a)
  const xaiItems    = (track.xai ?? []).filter(x => x.importance > 0.02).sort((a,b) => b.importance-a.importance).slice(0,8)

  return (
    <div ref={backdropRef} onClick={e => { if (e.target===backdropRef.current) onClose() }}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
      <div className="w-full max-w-2xl max-h-[92vh] overflow-y-auto rounded-2xl border-2 shadow-2xl"
        style={{ borderColor:(s?.color??'#888')+'80', backgroundColor:'#0a1628' }}>

        {/* Header */}
        <div className="sticky top-0 flex items-center gap-4 px-6 py-4 border-b border-white/10 z-10"
          style={{ backgroundColor: s?.bg ?? '#111' }}>
          <span className="text-5xl">{s?.icon}</span>
          <div className="flex-1">
            <p className="text-xs tracking-[0.2em] text-slate-400 mb-0.5">AI CLASSIFICATION — {track.track_id}</p>
            <p className="text-3xl font-bold font-mono tracking-wider" style={{ color:s?.color }}>{track.ai_class}</p>
          </div>
          <div className="text-right mr-4">
            <p className="text-xs text-slate-400 tracking-wider mb-0.5">CONFIDENCE</p>
            <p className="text-4xl font-bold font-mono" style={{ color:s?.color }}>{Math.round(track.ai_conf*100)}%</p>
          </div>
          <button onClick={onClose}
            className="text-slate-400 hover:text-white text-2xl w-9 h-9 flex items-center justify-center rounded-lg hover:bg-white/10 transition-colors">✕</button>
        </div>

        <div className="p-6 space-y-6">
          {/* Kinematic */}
          <section>
            <h4 className="text-xs font-bold text-cyan-400 tracking-[0.2em] mb-3">📊 KINEMATIC DATA</h4>
            <div className="grid grid-cols-3 gap-3">
              {[
                { label:'Altitude', val:`${track.altitude_ft.toLocaleString()} ft` },
                { label:'Speed',    val:`${Math.round(track.speed_kts)} kts` },
                { label:'Heading',  val:`${track.heading}°` },
                { label:'RCS',      val:`${track.rcs_m2} m²` },
                { label:'Profile',  val:track.flight_profile.replace(/_/g,' ') },
                { label:'Weather',  val:track.weather },
              ].map(({ label, val }) => (
                <div key={label} className="bg-slate-900/70 border border-slate-700/60 rounded-xl p-3">
                  <p className="text-xs text-slate-400 mb-1">{label}</p>
                  <p className="font-mono text-white font-semibold text-sm">{val}</p>
                </div>
              ))}
            </div>
          </section>

          {/* Electronic */}
          <section>
            <h4 className="text-xs font-bold text-orange-400 tracking-[0.2em] mb-3">📻 ELECTRONIC SIGNATURE</h4>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-slate-900/70 border border-slate-700/60 rounded-xl p-3">
                <p className="text-xs text-slate-400 mb-1">ESM — Passive Emissions</p>
                <p className="font-mono text-orange-300 font-bold text-sm">{track.esm_signature.replace(/_/g,' ')}</p>
              </div>
              <div className="bg-slate-900/70 border border-slate-700/60 rounded-xl p-3">
                <p className="text-xs text-slate-400 mb-1">IFF — Transponder Mode</p>
                <p className="font-mono text-blue-300 font-bold text-sm">{track.iff_mode.replace(/_/g,' ')}</p>
              </div>
            </div>
          </section>

          {/* Sensor Fusion */}
          <section>
            <h4 className="text-xs font-bold text-purple-400 tracking-[0.2em] mb-3">🔀 SENSOR FUSION</h4>
            <div className="space-y-2.5">
              {Object.values(track.sensor_votes ?? {}).map(sv => {
                const sc = CLASS_STYLES[sv.vote]
                return (
                  <div key={sv.label} className="flex items-center gap-3 bg-slate-900/50 rounded-xl px-4 py-3">
                    <span className="text-xl w-7">{sv.icon}</span>
                    <span className="text-slate-300 w-12 font-bold text-sm">{sv.label}</span>
                    <span className="text-slate-400 flex-1 font-mono text-xs">{sv.reading}</span>
                    <div className="w-28 h-2 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width:`${sv.conf*100}%`, backgroundColor:sc?.color??'#888' }}/>
                    </div>
                    <span className="font-mono font-bold text-sm w-20 text-right" style={{ color:sc?.color }}>
                      {sv.vote} {Math.round(sv.conf*100)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </section>

          {/* Class probabilities */}
          <section>
            <h4 className="text-xs font-bold text-slate-300 tracking-[0.2em] mb-3">📈 CLASS PROBABILITIES</h4>
            <div className="space-y-2">
              {sortedProbs.map(([cls, prob]) => {
                const cs = CLASS_STYLES[cls]
                return (
                  <div key={cls} className="flex items-center gap-3">
                    <span className="text-lg w-7 text-center">{cs?.icon}</span>
                    <span className="w-32 font-mono text-xs text-right" style={{ color:cs?.color }}>{cls}</span>
                    <div className="flex-1 h-2.5 bg-slate-800 rounded-full overflow-hidden">
                      <div className="h-full rounded-full" style={{ width:`${prob*100}%`, backgroundColor:cs?.color }}/>
                    </div>
                    <span className="w-10 font-mono font-bold text-sm text-right" style={{ color:cs?.color }}>
                      {(prob*100).toFixed(0)}%
                    </span>
                  </div>
                )
              })}
            </div>
          </section>

          {/* Anomaly Detection */}
          <section>
            <h4 className="text-xs font-bold text-amber-400 tracking-[0.2em] mb-3">⚠ ANOMALY DETECTION</h4>
            <AnomalyAlerts anomalies={track.anomalies ?? []}/>
          </section>

          {/* 3D Trajectory */}
          <section>
            <h4 className="text-xs font-bold text-cyan-400 tracking-[0.2em] mb-3">
              📈 FLIGHT TRAJECTORY — 3D ENVELOPE
              {!track.hist_alts && <span className="ml-2 text-slate-600 font-normal normal-case tracking-normal">(reconstructed)</span>}
            </h4>
            {/* Disable modal scroll while interacting with the 3D chart */}
            <div className="bg-slate-950/60 border border-slate-800/60 rounded-xl p-3"
              style={{ userSelect:'text' }}
              onMouseEnter={e => { (e.currentTarget.closest('.overflow-y-auto') as HTMLElement | null)?.style.setProperty('overflow-y','hidden') }}
              onMouseLeave={e => { (e.currentTarget.closest('.overflow-y-auto') as HTMLElement | null)?.style.setProperty('overflow-y','auto') }}
              onMouseDown={e => e.stopPropagation()}
              onPointerDown={e => e.stopPropagation()}
              onWheel={e => e.stopPropagation()}
            >
              <Trail3D track={simToTrailTrack(track)}/>
            </div>
          </section>

          {/* XAI */}
          <section>
            <h4 className="text-xs font-bold text-green-400 tracking-[0.2em] mb-3">🧠 EXPLAINABLE AI — WHY THIS CLASSIFICATION?</h4>
            {xaiItems.length > 0 ? (
              <>
                <div className="space-y-2.5">
                  {xaiItems.map(item => {
                    const barColor = item.direction==='supporting'?'#22c55e':item.direction==='conflicting'?'#ef4444':'#94a3b8'
                    const pct = Math.min(100, item.importance*100*8)
                    return (
                      <div key={item.feature} className="flex items-center gap-3 bg-slate-900/50 rounded-xl px-4 py-3">
                        <span className="text-base">{item.direction==='supporting'?'✅':item.direction==='conflicting'?'❌':'➖'}</span>
                        <div className="flex-1">
                          <div className="flex justify-between mb-1.5">
                            <span className="text-slate-200 font-semibold text-sm">{item.label}</span>
                            <span className="text-slate-400 font-mono text-xs">{item.value}</span>
                          </div>
                          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full rounded-full" style={{ width:`${pct}%`, backgroundColor:barColor }}/>
                          </div>
                        </div>
                        <span className="w-16 font-mono text-sm text-right font-bold" style={{ color:barColor }}>
                          {item.direction==='supporting'?'+':item.direction==='conflicting'?'-':''}
                          {(item.delta*100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  })}
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  Confidence shift when each feature is neutralised. ✅ supports · ❌ conflicts with classification.
                </p>
              </>
            ) : (
              <p className="text-sm text-slate-500 italic bg-slate-900/40 rounded-xl px-4 py-3">
                XAI data not available for demo tracks. Submit a live track to see feature importance analysis.
              </p>
            )}
          </section>

          {/* Footer */}
          <div className="pt-2 border-t border-white/10 flex items-center justify-between text-xs text-slate-500">
            <span>Submitted: <span className="text-slate-400">{track.submitted_at}</span></span>
            <span>Map: ({track.pos.x}, {track.pos.y})</span>
            <button onClick={onClose}
              className="text-slate-400 hover:text-white border border-slate-600 hover:border-slate-400 px-4 py-2 rounded-lg transition-colors text-sm">
              Close ✕
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Alert banner ──────────────────────────────────────────────────────────────
interface AlertEntry { track: SimTrack; id: number }

function AlertBanner({ alert, onInspect, onDismiss }: {
  alert: AlertEntry; onInspect: (t: SimTrack) => void; onDismiss: (id: number) => void
}) {
  const s = CLASS_STYLES[alert.track.ai_class]
  useEffect(() => {
    const t = setTimeout(() => onDismiss(alert.id), 7000)
    return () => clearTimeout(t)
  }, [alert.id, onDismiss])

  return (
    <div className="flex items-center gap-3 rounded-xl border px-4 py-3 text-sm shadow-2xl"
      style={{ borderColor:(s?.color??'#888')+'70', backgroundColor:s?.bg??'#111' }}>
      <span className="text-2xl animate-bounce">{s?.icon}</span>
      <div className="flex-1">
        <p className="text-xs text-slate-400 tracking-widest">NEW CONTACT DETECTED</p>
        <p className="font-bold font-mono text-sm" style={{ color:s?.color }}>{alert.track.ai_class}</p>
        <p className="text-xs text-slate-400">{alert.track.track_id} · {Math.round(alert.track.ai_conf*100)}% conf</p>
      </div>
      <button onClick={() => onInspect(alert.track)}
        className="px-3 py-2 rounded-lg text-xs font-bold transition-colors hover:opacity-90"
        style={{ backgroundColor:(s?.color??'#888')+'25', color:s?.color, border:`1px solid ${s?.color}60` }}>
        Inspect
      </button>
      <button onClick={() => onDismiss(alert.id)} className="text-slate-500 hover:text-white w-6 h-6 flex items-center justify-center">✕</button>
    </div>
  )
}

// ── Main ──────────────────────────────────────────────────────────────────────
type WsState  = 'connecting' | 'live' | 'disconnected'
type ViewMode = '2d' | '3d'

export default function SimDisplay() {
  const { sessionId } = useParams<{ sessionId: string }>()

  const [tracks,       setTracks]       = useState<SimTrack[]>(DEMO_TRACKS)
  const [participants, setParticipants] = useState(0)
  const [wsState,      setWsState]      = useState<WsState>('connecting')
  const [newIds,       setNewIds]       = useState<Set<string>>(new Set())
  const [copied,       setCopied]       = useState(false)
  const [alerts,       setAlerts]       = useState<AlertEntry[]>([])
  const [inspectTrack, setInspect]      = useState<SimTrack | null>(null)
  const [tick,         setTick]         = useState(0)
  const [viewMode,     setViewMode]     = useState<ViewMode>('2d')
  const alertCounter = useRef(0)

  // Animation loop
  useEffect(() => {
    const id = setInterval(() => setTick(t => t+1), 80)
    return () => clearInterval(id)
  }, [])

  // Load existing tracks
  useEffect(() => {
    if (!sessionId) return
    fetch(`${BASE}/sim/${sessionId}/tracks`)
      .then(r => r.json())
      .then(d => {
        if ((d.tracks??[]).length > 0) setTracks([...DEMO_TRACKS, ...(d.tracks??[])])
        setParticipants(d.participant_count ?? 0)
      })
      .catch(console.error)
  }, [sessionId])

  // WebSocket
  useEffect(() => {
    if (!sessionId) return
    const ws = new WebSocket(getWsUrl(sessionId))
    ws.onopen  = () => setWsState('live')
    ws.onclose = () => setWsState('disconnected')
    ws.onerror = () => setWsState('disconnected')
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data)
        if (msg.type === 'new_track') {
          const t: SimTrack = msg.track
          setTracks(prev => [...prev, t])
          setNewIds(prev => new Set([...prev, t.track_id]))
          setTimeout(() => setNewIds(prev => { const n=new Set(prev); n.delete(t.track_id); return n }), 2500)
          const alertId = ++alertCounter.current
          setAlerts(prev => [...prev.slice(-2), { track:t, id:alertId }])
        }
      } catch { /**/ }
    }
    return () => ws.close()
  }, [sessionId])

  const participantLink = `${window.location.origin}/sim/${sessionId}`
  const copyLink = useCallback(() => {
    navigator.clipboard.writeText(participantLink).then(() => { setCopied(true); setTimeout(()=>setCopied(false),2000) })
  }, [participantLink])
  const dismissAlert = useCallback((id: number) => setAlerts(p => p.filter(a => a.id !== id)), [])

  const DEMO_IDS      = new Set(['SIM-H001','SIM-S002','SIM-U003','SIM-N004','SIM-A005','SIM-F006'])
  const submittedCount = tracks.filter(t => !DEMO_IDS.has(t.track_id)).length
  const threats        = tracks.filter(t => t.ai_class==='HOSTILE'||t.ai_class==='SUSPECT').length
  const friendly       = tracks.filter(t => t.ai_class==='FRIEND'||t.ai_class==='ASSUMED FRIEND').length
  const wsColor        = wsState==='live'?'bg-green-400 animate-pulse':wsState==='connecting'?'bg-yellow-400':'bg-red-500'

  return (
    <div className="h-screen bg-[#060d19] text-white flex flex-col overflow-hidden select-none">

      {/* Top bar */}
      <div className="flex items-center gap-3 px-4 h-12 bg-black/40 border-b border-slate-800/80 shrink-0 text-xs">
        <span className="font-bold tracking-[0.15em] text-cyan-400 hidden sm:block">VANGUARD AI — TACTICAL MOC</span>
        <div className="flex items-center gap-1.5">
          <span className="text-slate-500">SESSION</span>
          <span className="font-mono font-bold text-yellow-300 tracking-widest text-sm">{sessionId}</span>
        </div>
        <button onClick={copyLink}
          className={`border rounded px-2 py-0.5 transition-colors ${copied?'border-green-500 text-green-400':'border-slate-700 text-slate-400 hover:border-slate-500 hover:text-white'}`}>
          {copied ? '✓ Copied!' : '📋 Copy Link'}
        </button>

        {/* ── View toggle ── */}
        <div className="flex items-center gap-0.5 bg-slate-900 border border-slate-700 rounded-lg p-0.5 ml-1">
          <button
            onClick={() => setViewMode('2d')}
            className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${
              viewMode==='2d'
                ? 'bg-cyan-600 text-white shadow'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >2D</button>
          <button
            onClick={() => setViewMode('3d')}
            className={`px-3 py-1 rounded-md text-xs font-bold transition-all ${
              viewMode==='3d'
                ? 'bg-cyan-600 text-white shadow'
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >3D</button>
        </div>

        <div className="ml-auto flex items-center gap-4 text-slate-400">
          <span className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${wsColor}`}/>
            <span className={wsState==='live'?'text-green-400':wsState==='connecting'?'text-yellow-400':'text-red-400'}>
              {wsState.toUpperCase()}
            </span>
          </span>
          <span>📡 {participants} online</span>
          <span className="text-slate-300 font-semibold">📤 {submittedCount} submitted</span>
          {threats>0  && <span className="text-red-400">⚠ {threats} threats</span>}
          {friendly>0 && <span className="text-green-400">🛡 {friendly} friendly</span>}
          <Link to="/sim" className="text-slate-600 hover:text-slate-400 transition-colors ml-1">← Home</Link>
        </div>
      </div>

      {/* Main */}
      <div className="flex flex-1 overflow-hidden relative">

        {/* Map area */}
        <div className="flex-1 flex items-stretch overflow-hidden relative">

          {/* ── 2D SVG MAP ── */}
          {viewMode === '2d' && (
            <svg viewBox="0 0 1000 600" preserveAspectRatio="xMidYMid meet" className="w-full h-full">
              <defs>
                <pattern id="seaWaves" width="50" height="18" patternUnits="userSpaceOnUse" patternTransform="rotate(-4)">
                  <path d="M -10,9 Q 2.5,3 15,9 Q 27.5,15 40,9" fill="none" stroke="rgba(0,80,170,0.07)" strokeWidth="1.2"/>
                </pattern>
                <pattern id="enemyHatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="12" stroke="rgba(120,20,20,0.12)" strokeWidth="1"/>
                </pattern>
                <pattern id="friendHatch" width="12" height="12" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
                  <line x1="0" y1="0" x2="0" y2="12" stroke="rgba(20,50,120,0.12)" strokeWidth="1"/>
                </pattern>
                <radialGradient id="shipGlow" cx="50%" cy="50%" r="50%">
                  <stop offset="0%"   stopColor="#00ffff" stopOpacity="0.28"/>
                  <stop offset="100%" stopColor="#00ffff" stopOpacity="0"/>
                </radialGradient>
                <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
                  <feGaussianBlur stdDeviation="3" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <filter id="trackGlow" x="-80%" y="-80%" width="260%" height="260%">
                  <feGaussianBlur stdDeviation="5" result="blur"/>
                  <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
              </defs>

              <rect width="1000" height="600" fill="#060d19"/>

              {/* Enemy territory */}
              <path
                d="M 0,0 L 235,0 C 258,22 244,50 212,74 C 194,90 232,112 208,138
                   C 188,160 228,178 205,208 C 184,234 226,252 200,282
                   C 176,308 220,326 196,358 C 174,386 222,404 196,436
                   C 175,462 222,480 195,512 C 172,536 218,556 190,582
                   C 175,594 208,600 188,600 L 0,600 Z"
                fill="rgba(80,15,15,0.55)" stroke="rgba(180,40,40,0.28)" strokeWidth="1.5"
              />
              <path
                d="M 0,0 L 235,0 C 258,22 244,50 212,74 C 194,90 232,112 208,138
                   C 188,160 228,178 205,208 C 184,234 226,252 200,282
                   C 176,308 220,326 196,358 C 174,386 222,404 196,436
                   C 175,462 222,480 195,512 C 172,536 218,556 190,582
                   C 175,594 208,600 188,600 L 0,600 Z"
                fill="url(#enemyHatch)"
              />
              <g fill="rgba(140,30,30,0.35)" stroke="rgba(160,40,40,0.2)" strokeWidth="0.5">
                <polygon points="65,155 85,118 105,155"/>
                <polygon points="50,185 75,142 100,185"/>
                <polygon points="80,210 102,172 124,210"/>
                <polygon points="55,240 78,200 101,240"/>
                <polygon points="75,275 95,240 115,275"/>
              </g>
              <path d="M 100,60 C 115,120 85,180 105,240 C 125,300 90,360 110,420"
                fill="none" stroke="rgba(0,100,180,0.25)" strokeWidth="1.5"/>
              <text x="88"  y="42"  fill="rgba(220,60,60,0.55)"  fontSize="9" fontFamily="monospace" fontWeight="bold" letterSpacing="2">ENEMY TERRITORY</text>
              <text x="72"  y="340" fill="rgba(180,50,50,0.38)"  fontSize="8" fontFamily="monospace">⬡ KHRAMI AB</text>
              <text x="55"  y="480" fill="rgba(180,50,50,0.35)"  fontSize="7" fontFamily="monospace">⬡ PORT SEVERNY</text>

              {/* Friendly territory */}
              <path
                d="M 1000,0 L 765,0 C 742,22 756,50 788,74 C 806,90 768,112 792,138
                   C 812,160 772,178 795,208 C 816,234 774,252 800,282
                   C 824,308 780,326 804,358 C 826,386 778,404 804,436
                   C 825,462 778,480 805,512 C 828,536 782,556 810,582
                   C 825,594 792,600 812,600 L 1000,600 Z"
                fill="rgba(10,30,80,0.55)" stroke="rgba(40,80,200,0.28)" strokeWidth="1.5"
              />
              <path
                d="M 1000,0 L 765,0 C 742,22 756,50 788,74 C 806,90 768,112 792,138
                   C 812,160 772,178 795,208 C 816,234 774,252 800,282
                   C 824,308 780,326 804,358 C 826,386 778,404 804,436
                   C 825,462 778,480 805,512 C 828,536 782,556 810,582
                   C 825,594 792,600 812,600 L 1000,600 Z"
                fill="url(#friendHatch)"
              />
              <g fill="rgba(20,50,120,0.30)" stroke="rgba(30,70,160,0.2)" strokeWidth="0.5">
                <polygon points="895,165 915,128 935,165"/>
                <polygon points="910,195 930,158 950,195"/>
                <polygon points="870,230 894,192 918,230"/>
                <polygon points="900,262 922,224 944,262"/>
              </g>
              <path d="M 900,55 C 885,115 915,175 895,235 C 875,295 910,355 890,415"
                fill="none" stroke="rgba(0,100,200,0.22)" strokeWidth="1.5"/>
              <text x="820" y="42"  fill="rgba(60,120,220,0.55)"  fontSize="9" fontFamily="monospace" fontWeight="bold" letterSpacing="2">FRIENDLY TERRITORY</text>
              <text x="825" y="340" fill="rgba(50,100,200,0.38)"  fontSize="8" fontFamily="monospace">⬡ INCIRLIK AB</text>
              <text x="835" y="480" fill="rgba(50,100,200,0.35)"  fontSize="7" fontFamily="monospace">⬡ PORT AEGEAN</text>

              <rect x="200" y="0" width="600" height="600" fill="url(#seaWaves)"/>
              <text x="500" y="580" textAnchor="middle" fill="rgba(0,130,200,0.15)" fontSize="12" fontFamily="monospace" letterSpacing="5">NEUTRAL SEA ZONE</text>

              {/* Grid */}
              {[100,200,300,400,600,700,800,900].map(x => (
                <line key={x} x1={x} y1="0" x2={x} y2="600" stroke="rgba(0,150,220,0.04)" strokeWidth="1"/>
              ))}
              {[100,200,400,500].map(y => (
                <line key={y} x1="0" y1={y} x2="1000" y2={y} stroke="rgba(0,150,220,0.04)" strokeWidth="1"/>
              ))}

              <line x1="465" y1="0" x2="465" y2="600" stroke="rgba(200,50,50,0.14)"  strokeWidth="1" strokeDasharray="6,5"/>
              <line x1="535" y1="0" x2="535" y2="600" stroke="rgba(50,100,220,0.14)" strokeWidth="1" strokeDasharray="6,5"/>

              {SPOKES.map(({ x2, y2 }, i) => (
                <line key={i} x1="500" y1="300" x2={x2} y2={y2} stroke="rgba(0,150,220,0.05)" strokeWidth="1"/>
              ))}
              {[100,200,300].map(r => (
                <circle key={r} cx="500" cy="300" r={r} fill="none" stroke="rgba(0,150,220,0.12)" strokeWidth="1" strokeDasharray="4,10"/>
              ))}
              {[{r:100,l:'20NM'},{r:200,l:'40NM'},{r:300,l:'60NM'}].map(({r,l}) => (
                <text key={r} x={500+r+5} y={297} fill="rgba(0,150,220,0.35)" fontSize="9" fontFamily="monospace">{l}</text>
              ))}

              <text x="500" y="18"  textAnchor="middle" fill="rgba(0,200,255,0.45)" fontSize="10" fontFamily="monospace" fontWeight="bold">N</text>
              <text x="500" y="595" textAnchor="middle" fill="rgba(0,200,255,0.45)" fontSize="10" fontFamily="monospace" fontWeight="bold">S</text>
              <text x="988" y="304" textAnchor="middle" fill="rgba(0,200,255,0.45)" fontSize="10" fontFamily="monospace" fontWeight="bold">E</text>
              <text x="14"  y="304" textAnchor="middle" fill="rgba(0,200,255,0.45)" fontSize="10" fontFamily="monospace" fontWeight="bold">W</text>

              <text x="170" y="26" textAnchor="middle" fill="rgba(210,50,50,0.60)"  fontSize="10" fontFamily="monospace" letterSpacing="3" fontWeight="bold">THREAT AXIS</text>
              <text x="830" y="26" textAnchor="middle" fill="rgba(50,110,210,0.60)" fontSize="10" fontFamily="monospace" letterSpacing="3" fontWeight="bold">FRIENDLY AXIS</text>

              {/* Commercial airways */}
              {COM_ROUTES.map(r => (
                <line key={r.id} x1={r.x1} y1={r.y1} x2={r.x2} y2={r.y2}
                  stroke="rgba(200,220,255,0.13)" strokeWidth="1.5" strokeDasharray="10,6"/>
              ))}
              <text x="520" y="84"  fill="rgba(200,220,255,0.30)" fontSize="9" fontFamily="monospace" fontWeight="bold">N600</text>
              <text x="520" y="220" fill="rgba(200,220,255,0.30)" fontSize="9" fontFamily="monospace" fontWeight="bold">B57</text>
              <text x="520" y="490" fill="rgba(200,220,255,0.30)" fontSize="9" fontFamily="monospace" fontWeight="bold">UL613</text>

              {/* Moving commercial aircraft */}
              {COM_AC.map(ac => {
                const { x, y, angle } = comPos(ac, tick)
                return (
                  <g key={ac.id} transform={`translate(${x},${y})`} opacity="0.7">
                    <g transform={`rotate(${angle + 90})`}>
                      <ellipse rx="2.5" ry="9" fill="rgba(220,235,255,0.75)"/>
                      <polygon points="0,-2 -11,3 -6,5 0,2 6,5 11,3" fill="rgba(200,215,255,0.65)"/>
                      <polygon points="0,6 -5,10 5,10" fill="rgba(200,215,255,0.60)"/>
                    </g>
                    <text y="18" textAnchor="middle" fill="rgba(180,210,255,0.55)" fontSize="7.5" fontFamily="monospace">{ac.lbl}</text>
                    <text y="26" textAnchor="middle" fill="rgba(140,170,220,0.42)" fontSize="6.5" fontFamily="monospace">{ac.fl}</text>
                  </g>
                )
              })}

              {/* Own ship */}
              <circle cx="500" cy="300" r="32" fill="url(#shipGlow)"/>
              <g transform="translate(500,300)" filter="url(#glow)">
                <polygon points="0,-12 12,0 0,12 -12,0" fill="rgba(0,240,240,0.75)" stroke="cyan" strokeWidth="1.5"/>
                <text y="24" textAnchor="middle" fill="rgba(0,240,240,0.65)" fontSize="8" fontFamily="monospace" letterSpacing="1">OWN SHIP</text>
              </g>

              {/* Animated track symbols */}
              {tracks.map(track => {
                const { sx, sy } = animPos(track, tick)
                const isNew  = newIds.has(track.track_id)
                const isDemo = track.submitted_at === 'DEMO'
                const color  = CLASS_STYLES[track.ai_class]?.color ?? '#888'
                const { dx, dy } = hdgVec(track.heading)
                return (
                  <g key={track.track_id}
                     transform={`translate(${sx},${sy})`}
                     filter={isNew ? 'url(#trackGlow)' : undefined}
                     opacity={isDemo ? 0.80 : 1}
                     style={{ cursor:'pointer' }}
                     onClick={() => setInspect(track)}
                  >
                    <line x1="0" y1="0" x2={dx} y2={dy} stroke={color} strokeWidth="1.5" opacity="0.6" strokeDasharray="3,2"/>
                    {isNew && (
                      <circle r="10" fill="none" stroke={color} strokeWidth="1.5" opacity="0">
                        <animate attributeName="r"       values="10;28;10" dur="1.8s" repeatCount="2"/>
                        <animate attributeName="opacity" values="0.9;0;0.9" dur="1.8s" repeatCount="2"/>
                      </circle>
                    )}
                    {isDemo && <circle r="13" fill="none" stroke={color} strokeWidth="0.6" opacity="0.25" strokeDasharray="2,4"/>}
                    <TrackSymbol cls={track.ai_class}/>
                    <text y="20" textAnchor="middle" fill={color} fontSize="8" fontFamily="monospace" fontWeight="bold">{track.track_id}</text>
                    <text y="29" textAnchor="middle" fill={color} fontSize="7" fontFamily="monospace" opacity="0.70">{Math.round(track.ai_conf*100)}%</text>
                  </g>
                )
              })}
            </svg>
          )}

          {/* ── 3D VIEW ── */}
          {viewMode === '3d' && (
            <TacticalMap3D tracks={tracks} tick={tick} onInspect={setInspect}/>
          )}

          {/* Alert overlay */}
          <div className="absolute bottom-4 left-4 flex flex-col gap-2 max-w-xs z-30 pointer-events-none">
            {alerts.map(alert => (
              <div key={alert.id} className="pointer-events-auto">
                <AlertBanner alert={alert}
                  onInspect={t => { setInspect(t); dismissAlert(alert.id) }}
                  onDismiss={dismissAlert}/>
              </div>
            ))}
          </div>
        </div>

        {/* Sidebar */}
        <div className="w-72 bg-slate-950/60 border-l border-slate-800/70 flex flex-col overflow-hidden shrink-0">
          <div className="px-3 py-2.5 border-b border-slate-800/70 flex items-center justify-between">
            <span className="text-xs font-bold text-slate-400 tracking-[0.2em]">TRACK FEED</span>
            <div className="flex items-center gap-2 text-xs text-slate-600">
              <span>{tracks.length} total</span><span>·</span><span>{DEMO_TRACKS.length} demo</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-2 space-y-1.5">
            {[...tracks].reverse().map(track => {
              const s      = CLASS_STYLES[track.ai_class]
              const isDemo = track.submitted_at === 'DEMO'
              return (
                <button key={track.track_id} onClick={() => setInspect(track)}
                  className="w-full text-left rounded-xl p-3 border transition-all hover:brightness-125 active:scale-[0.98]"
                  style={{ borderColor:(s?.color??'#888')+'35', backgroundColor:(s?.bg??'#111')+'dd' }}>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-bold font-mono text-sm" style={{ color:s?.color }}>{s?.icon} {track.ai_class}</span>
                    <span className="text-xs text-slate-600">{isDemo ? '— DEMO —' : track.submitted_at}</span>
                  </div>
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-xs text-slate-500">{track.track_id}</span>
                    <span className="font-bold font-mono text-sm" style={{ color:s?.color }}>{Math.round(track.ai_conf*100)}%</span>
                  </div>
                  <div className="h-1 bg-slate-800 rounded-full overflow-hidden mb-1.5">
                    <div className="h-full rounded-full" style={{ width:`${track.ai_conf*100}%`, backgroundColor:s?.color }}/>
                  </div>
                  <div className="text-xs text-slate-500">{track.altitude_ft.toLocaleString()} ft · {Math.round(track.speed_kts)} kts</div>
                  <div className="text-xs text-slate-600 mt-0.5">ESM: {track.esm_signature.replace(/_/g,' ')}</div>
                  <div className="text-xs text-slate-700 mt-0.5">Tap to inspect ↗</div>
                </button>
              )
            })}
          </div>

          <div className="border-t border-slate-800/70 px-3 py-2">
            <p className="text-xs text-slate-700">6 demo tracks pre-loaded. Click any track to inspect.</p>
          </div>
        </div>
      </div>

      {inspectTrack && <InspectModal track={inspectTrack} onClose={() => setInspect(null)}/>}
    </div>
  )
}

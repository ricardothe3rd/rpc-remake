import React, { useState, useEffect, useCallback } from 'react'
import { io, Socket } from 'socket.io-client'

interface BatteryData {
  type: string
  level: number
  timestamp?: string
}

interface WifiData {
  type: string
  rssi_dbm: number
  signal_strength?: string
  timestamp?: string
}

interface CmdVelData {
  type: string
  linear: {
    x: number
    y: number
    z: number
  }
  angular: {
    x: number
    y: number
    z: number
  }
  status?: string
  timestamp?: string
}

interface MapData {
  type: string
  header: {
    stamp: { sec: number; nanosec: number }
    frame_id: string
  }
  info: {
    map_load_time: { sec: number; nanosec: number }
    resolution: number
    width: number
    height: number
    origin: {
      position: { x: number; y: number; z: number }
      orientation: { x: number; y: number; z: number; w: number }
    }
  }
  data: number[]
  timestamp?: string
}

interface LaserScanData {
  type: string
  header: {
    stamp: { sec: number; nanosec: number }
    frame_id: string
  }
  angle_min: number
  angle_max: number
  angle_increment: number
  time_increment: number
  scan_time: number
  range_min: number
  range_max: number
  ranges: number[]
  intensities: number[]
  timestamp?: string
}

interface CameraData {
  type: string
  header: {
    stamp: { sec: number; nanosec: number }
    frame_id: string
  }
  format: string
  data: string
  timestamp?: string
}

interface NavigationStatus {
  type: string
  status: string
  message: string
  timestamp?: string
}

interface NavigationFeedback {
  type: string
  distance_remaining: number
  estimated_time_remaining: number
  current_pose: {
    x: number
    y: number
    z: number
    qx: number
    qy: number
    qz: number
    qw: number
  }
  timestamp?: string
}

interface RobotPose {
  type: string
  x: number
  y: number
  yaw: number  // yaw in degrees
  frame_id: string
  timestamp?: string
}

interface RobotSpecification {
  type: string
  api_version: number
  model_name: string
  manufacturer: string
  firmware_version: string
  hardware_version: string
  sensors: string[]
  capabilities: string[]
  battery_capacity_mah: number
  max_speed_ms: number
  shape: {
    type: string
    diameter_m: number
  }
  height_m: number
  lidar_sensor?: {
    position: { x_m: number; y_m: number; z_m: number }
    orientation: { roll_deg: number; pitch_deg: number; yaw_deg: number }
    min_range_m: number
    max_range_m: number
  }
  drive_type: string
  wheel_track_distance_m?: number
  docking_capability?: boolean
  max_angular_speed_rad_s?: number
  weight_kg?: number
  timestamp?: string
}


interface DetectedObject {
  type: string
  distance: number
  angle_rad: number
  angle_deg: number
  x: number
  y: number
  confidence: number
  timestamp: string
}

interface GeometricFeature {
  type: string  // 'line', 'circle', 'corner'
  points: [number, number][]  // array of [x, y] coordinates
  confidence: number
  parameters: { [key: string]: any }
}

interface ObjectDetectionData {
  objects: DetectedObject[]
  summary: {
    total_objects: number
    object_counts: { [key: string]: number }
    last_update: string | null
    scan_available: boolean
  }
  features: GeometricFeature[]
}

interface ConsoleMessage {
  type: string
  content: string
  timestamp: number
}

const App: React.FC = () => {
  // Read URL parameters (passed from AppStore)
  const params = new URLSearchParams(window.location.search)
  const sessionId = params.get('session_id')
  const robotId = params.get('robot_id')
  const token = params.get('token')
  const apiUrl = params.get('api_url') || 'http://localhost:5000'

  const [robotConnected, setRobotConnected] = useState(false)
  const [batteryData, setBatteryData] = useState<BatteryData | null>(null)
  const [wifiData, setWifiData] = useState<WifiData | null>(null)
  const [cmdVelData, setCmdVelData] = useState<CmdVelData | null>(null)
  const [mapData, setMapData] = useState<MapData | null>(null)
  const [scanData, setScanData] = useState<LaserScanData | null>(null)
  const [cameraData, setCameraData] = useState<CameraData | null>(null)
  const [navigationStatus, setNavigationStatus] = useState<NavigationStatus | null>(null)
  const [navigationFeedback, setNavigationFeedback] = useState<NavigationFeedback | null>(null)
  const [robotPose, setRobotPose] = useState<RobotPose | null>(null)
  const [robotSpec, setRobotSpec] = useState<RobotSpecification | null>(null)
  const [linearX, setLinearX] = useState(0)
  const [angularZ, setAngularZ] = useState(0)
  const [socket, setSocket] = useState<Socket | null>(null)

  // Speed acceleration state
  const [currentLinearSpeed, setCurrentLinearSpeed] = useState(0)
  const [currentAngularSpeed, setCurrentAngularSpeed] = useState(0)
  const BASE_LINEAR_SPEED = 0.1
  const BASE_ANGULAR_SPEED = 0.3
  const LINEAR_INCREMENT = 0.1
  const ANGULAR_INCREMENT = 0.2
  const MAX_LINEAR_SPEED = 1.0
  const MAX_ANGULAR_SPEED = 2.0
  const [gridWidth, setGridWidth] = useState(384)
  const [gridHeight, setGridHeight] = useState(384)
  const [gridResolution, setGridResolution] = useState(0.05)
  const [generatedGrid, setGeneratedGrid] = useState<number[] | null>(null)
  const [navGoalX, setNavGoalX] = useState(0)
  const [navGoalY, setNavGoalY] = useState(0)
  const [navGoalYaw, setNavGoalYaw] = useState(0)
  const [navGoalRelative, setNavGoalRelative] = useState(false)
  const [objectDetectionData, setObjectDetectionData] = useState<ObjectDetectionData | null>(null)
  const [consoleMessages, setConsoleMessages] = useState<ConsoleMessage[]>([])
  const [lastDataUpdate, setLastDataUpdate] = useState<string | null>(null)
  const [connectionError, setConnectionError] = useState<string | null>(null)

  // Helper function to update last data timestamp
  const updateLastDataTimestamp = useCallback((timestamp?: string) => {
    if (timestamp) {
      setLastDataUpdate(timestamp)
    }
  }, [])

  // Helper function to decode binary ranges from base64
  const decodeBinaryRanges = (base64String: string): number[] => {
    const binaryString = atob(base64String)
    const bytes = new Uint8Array(binaryString.length)
    for (let i = 0; i < binaryString.length; i++) {
      bytes[i] = binaryString.charCodeAt(i)
    }
    const floats = new Float32Array(bytes.buffer)
    return Array.from(floats)
  }

  const connectWebSocket = useCallback(() => {
    // Check if URL parameters are present
    if (!sessionId || !robotId || !token) {
      setConnectionError('Missing required URL parameters (session_id, robot_id, token)')
      console.error('Missing URL parameters:', { sessionId, robotId, token, apiUrl })
      return
    }

    console.log('Connecting to AppStore WebSocket:', { apiUrl, sessionId, robotId })

    // Connect to AppStore WebSocket (not RPC backend)
    const newSocket = io(`${apiUrl}/sessions/${sessionId}/robot`, {
      auth: { token, robot_id: robotId }
    })

    newSocket.on('connect', () => {
      console.log('Connected to AppStore WebSocket')
      setSocket(newSocket)
      setConnectionError(null)
    })

    // Set up event handlers for robot data
    newSocket.on('battery', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setBatteryData(data)
    })

    newSocket.on('robot_pose', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setRobotPose(data)
    })

    newSocket.on('laser_scan', (data) => {
      updateLastDataTimestamp(data.timestamp)
      // Decode binary ranges if present (new format)
      if (data.ranges_binary && data.ranges_count) {
        data.ranges = decodeBinaryRanges(data.ranges_binary)
        delete data.ranges_binary
        delete data.ranges_count
      }
      setScanData(data)
    })

    newSocket.on('camera', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setCameraData(data)
    })

    newSocket.on('wifi', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setWifiData(data)
    })

    newSocket.on('cmd_vel', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setCmdVelData(data)
    })

    newSocket.on('map', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setMapData(data)
    })

    newSocket.on('navigation_status', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setNavigationStatus(data)
    })

    newSocket.on('navigation_feedback', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setNavigationFeedback(data)
    })

    newSocket.on('object_detection', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setObjectDetectionData(data)
    })

    newSocket.on('robot_spec', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setRobotSpec(data)
    })

    newSocket.on('robot_specification', (data) => {
      updateLastDataTimestamp(data.timestamp)
      setRobotSpec(data)
    })

    newSocket.on('console', (data) => {
      setConsoleMessages(prev => {
        const newMessages = [...prev, data]
        // Keep only the last 100 messages
        return newMessages.slice(-100)
      })
    })

    newSocket.on('connected', () => {
      console.log('Robot connected')
      setRobotConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Socket disconnected')
      setRobotConnected(false)
      setSocket(null)
    })

    newSocket.on('error', (err) => {
      console.error('Socket error:', err)
      setConnectionError(`Socket error: ${err.message || err}`)
    })

    newSocket.on('connect_error', (err) => {
      console.error('Connection error:', err)
      setConnectionError(`Connection error: ${err.message || err}`)
    })

    return newSocket
  }, [sessionId, robotId, token, apiUrl, updateLastDataTimestamp])

  useEffect(() => {
    const newSocket = connectWebSocket()
    return () => {
      if (newSocket) {
        newSocket.close()
      }
    }
  }, [connectWebSocket])

  // Initialize chat session with RPC backend
  useEffect(() => {
    if (!sessionId || !token || !apiUrl) {
      console.log('Skipping init-session: missing params')
      return
    }

    const initSession = async () => {
      try {
        console.log('Initializing chat session with RPC backend...')
        const response = await fetch(`${window.location.origin}/api/init-session`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            session_id: sessionId,
            session_token: token,
            appstore_url: apiUrl,
          }),
        })
        const data = await response.json()
        console.log('Init session response:', data)
      } catch (error) {
        console.error('Failed to initialize chat session:', error)
      }
    }

    initSession()
  }, [sessionId, token, apiUrl])

  // Robot status is now managed via Socket.IO events (connected/disconnect)
  // No need to poll the API anymore

  // Object detection data is now received via Socket.IO events
  // No need to poll the API anymore


  const sendTwistCommand = (linear_x = linearX, angular_z = angularZ) => {
    if (socket && socket.connected) {
      socket.emit('twist_command', { linear_x, angular_z })
      console.log('Twist command sent via Socket.IO:', { linear_x, angular_z })
    } else {
      console.error('Socket not connected')
    }
  }

  // Acceleration functions for Quick Twist Commands
  const accelerateForward = () => {
    const newSpeed = currentLinearSpeed <= 0
      ? BASE_LINEAR_SPEED
      : Math.min(currentLinearSpeed + LINEAR_INCREMENT, MAX_LINEAR_SPEED)
    setCurrentLinearSpeed(newSpeed)
    setCurrentAngularSpeed(0)
    setLinearX(newSpeed)
    setAngularZ(0)
    sendTwistCommand(newSpeed, 0)
  }

  const accelerateBackward = () => {
    const newSpeed = currentLinearSpeed >= 0
      ? -BASE_LINEAR_SPEED
      : Math.max(currentLinearSpeed - LINEAR_INCREMENT, -MAX_LINEAR_SPEED)
    setCurrentLinearSpeed(newSpeed)
    setCurrentAngularSpeed(0)
    setLinearX(newSpeed)
    setAngularZ(0)
    sendTwistCommand(newSpeed, 0)
  }

  const accelerateRotateRight = () => {
    const newSpeed = currentAngularSpeed <= 0
      ? BASE_ANGULAR_SPEED
      : Math.min(currentAngularSpeed + ANGULAR_INCREMENT, MAX_ANGULAR_SPEED)
    setCurrentAngularSpeed(newSpeed)
    setCurrentLinearSpeed(0)
    setLinearX(0)
    setAngularZ(newSpeed)
    sendTwistCommand(0, newSpeed)
  }

  const accelerateRotateLeft = () => {
    const newSpeed = currentAngularSpeed >= 0
      ? -BASE_ANGULAR_SPEED
      : Math.max(currentAngularSpeed - ANGULAR_INCREMENT, -MAX_ANGULAR_SPEED)
    setCurrentAngularSpeed(newSpeed)
    setCurrentLinearSpeed(0)
    setLinearX(0)
    setAngularZ(newSpeed)
    sendTwistCommand(0, newSpeed)
  }

  const stopRobot = () => {
    setCurrentLinearSpeed(0)
    setCurrentAngularSpeed(0)
    setLinearX(0)
    setAngularZ(0)
    sendTwistCommand(0, 0)
  }

  const sendTestCommand = () => {
    if (socket && socket.connected) {
      socket.emit('test_command', {})
      console.log('Test command sent via Socket.IO')
    } else {
      console.error('Socket not connected')
    }
  }

  const sendTestWSCommand = () => {
    if (socket && socket.connected) {
      socket.emit('test_ws', { content: 'Test Socket.IO message from UI' })
      console.log('Test command sent via Socket.IO')
    } else {
      console.error('Socket not connected')
    }
  }

  const stopApp = () => {
    if (socket && socket.connected) {
      socket.emit('stop_app', {})
      console.log('Stop app command sent via Socket.IO')
    } else {
      console.error('Socket not connected')
    }
  }


  const sendNavigationGoal = () => {
    if (!socket || !socket.connected) {
      console.error('Socket not connected')
      return
    }

    try {
      // Convert yaw to quaternion
      const yawRad = (navGoalYaw * Math.PI) / 180
      const qz = Math.sin(yawRad / 2)
      const qw = Math.cos(yawRad / 2)

      socket.emit('navigate_to_pose', {
        x: navGoalX,
        y: navGoalY,
        z: 0.0,
        qx: 0.0,
        qy: 0.0,
        qz: qz,
        qw: qw,
        frame_id: 'map',
        relative: navGoalRelative
      })
      console.log('Navigation goal sent via Socket.IO:', { x: navGoalX, y: navGoalY, yaw: navGoalYaw, relative: navGoalRelative })
    } catch (error) {
      console.error('Error sending navigation goal:', error)
    }
  }

  const cancelNavigation = () => {
    if (!socket || !socket.connected) {
      console.error('Socket not connected')
      return
    }

    try {
      socket.emit('cancel_navigation', {})
      console.log('Cancel navigation sent via Socket.IO')
    } catch (error) {
      console.error('Error cancelling navigation:', error)
    }
  }


  const generateOccupancyGrid = () => {
    const totalCells = gridWidth * gridHeight
    const grid: number[] = new Array(totalCells)

    // Fill with unknown cells first (-1)
    for (let i = 0; i < totalCells; i++) {
      grid[i] = -1
    }

    // Create some interesting patterns
    const centerX = gridWidth / 2
    const centerY = gridHeight / 2

    // Add free space in center area
    for (let y = Math.floor(centerY - 50); y < Math.floor(centerY + 50); y++) {
      for (let x = Math.floor(centerX - 50); x < Math.floor(centerX + 50); x++) {
        if (x >= 0 && x < gridWidth && y >= 0 && y < gridHeight) {
          const index = y * gridWidth + x
          grid[index] = 0 // Free space
        }
      }
    }

    // Add some obstacles (walls/barriers)
    // Vertical wall
    for (let y = Math.floor(centerY - 30); y < Math.floor(centerY + 30); y++) {
      const x = Math.floor(centerX + 20)
      if (x >= 0 && x < gridWidth && y >= 0 && y < gridHeight) {
        const index = y * gridWidth + x
        grid[index] = 100 // Occupied
      }
    }

    // Horizontal wall
    for (let x = Math.floor(centerX - 40); x < Math.floor(centerX); x++) {
      const y = Math.floor(centerY - 20)
      if (x >= 0 && x < gridWidth && y >= 0 && y < gridHeight) {
        const index = y * gridWidth + x
        grid[index] = 100 // Occupied
      }
    }

    // Add some random obstacles
    for (let i = 0; i < 5; i++) {
      const obstacleX = Math.floor(Math.random() * (gridWidth - 20)) + 10
      const obstacleY = Math.floor(Math.random() * (gridHeight - 20)) + 10
      const size = Math.floor(Math.random() * 10) + 5

      for (let dy = 0; dy < size; dy++) {
        for (let dx = 0; dx < size; dx++) {
          const x = obstacleX + dx
          const y = obstacleY + dy
          if (x < gridWidth && y < gridHeight) {
            const index = y * gridWidth + x
            grid[index] = 100 // Occupied
          }
        }
      }
    }

    setGeneratedGrid(grid)
    console.log(`Generated ${gridWidth}x${gridHeight} occupancy grid with ${totalCells} cells`)
  }

  const sendOccupancyGrid = () => {
    if (!generatedGrid) {
      console.error('No grid generated yet')
      return
    }

    if (!socket || !socket.connected) {
      console.error('Socket not connected')
      return
    }

    try {
      const gridData = {
        frame_id: 'map',
        resolution: gridResolution,
        width: gridWidth,
        height: gridHeight,
        origin: {
          x: -(gridWidth * gridResolution) / 2,
          y: -(gridHeight * gridResolution) / 2,
          z: 0.0,
          qx: 0.0,
          qy: 0.0,
          qz: 0.0,
          qw: 1.0
        },
        data: generatedGrid
      }

      socket.emit('occupancy_grid', { grid_data: gridData })
      console.log('OccupancyGrid command sent via Socket.IO')
    } catch (error) {
      console.error('Error sending OccupancyGrid command:', error)
    }
  }

  const renderLaserScan = () => {
    if (!scanData || !scanData.ranges) {
      return <p>No scan data available</p>
    }

    const width = 300
    const height = 300
    const centerX = width / 2
    const centerY = height / 2
    const maxRange = Math.min(scanData.range_max, 10) // Limit display range to 10 meters
    const scale = 3 * ((Math.min(width, height) / 2 - 20) / maxRange) // 3x scale factor

    const points: string[] = []

    // Convert polar coordinates to cartesian for visualization
    scanData.ranges.forEach((range, index) => {
      if (range >= scanData.range_min && range <= scanData.range_max && !isNaN(range) && isFinite(range)) {
        const angle = scanData.angle_min + index * scanData.angle_increment
        const displayRange = Math.min(range, maxRange)

        const x = centerX + displayRange * Math.cos(angle) * scale
        const y = centerY - displayRange * Math.sin(angle) * scale // Flip Y axis

        points.push(`${x},${y}`)
      }
    })

    // Get robot diameter from specification, default to 0.35m if not available
    const robotDiameterMeters = robotSpec?.shape?.diameter_m ?? 0.35
    const robotRadiusPixels = (robotDiameterMeters / 2) * scale
    const arrowLength = robotRadiusPixels * 1.5

    // Robot orientation (0 degrees = pointing right in scan visualization)
    const robotYawRad = 0 // Robot is at center, facing forward (0 rad)

    // Helper function to convert world coordinates to screen coordinates
    const worldToScreen = (x: number, y: number) => ({
      x: centerX + x * scale,
      y: centerY - y * scale  // Flip Y axis
    })

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <svg width={width} height={height} style={{ border: '1px solid #ccc', backgroundColor: '#f8f8f8' }}>
          {/* Draw grid circles */}
          {[1, 2, 3, 4, 5].map(radius => (
            <circle
              key={radius}
              cx={centerX}
              cy={centerY}
              r={radius * scale}
              fill="none"
              stroke="#ddd"
              strokeWidth="1"
            />
          ))}

          {/* Draw laser scan points */}
          {points.map((point, index) => {
            const [x, y] = point.split(',').map(Number)
            return (
              <circle
                key={index}
                cx={x}
                cy={y}
                r="1"
                fill="blue"
              />
            )
          })}

          {/* Draw detected geometric features */}
          {objectDetectionData?.features && objectDetectionData.features.map((feature, idx) => {
            if (feature.type === 'line') {
              // Draw line feature
              const screenPoints = feature.points.map(p => worldToScreen(p[0], p[1]))
              return (
                <g key={`line-${idx}`}>
                  <polyline
                    points={screenPoints.map(p => `${p.x},${p.y}`).join(' ')}
                    fill="none"
                    stroke="#00FF00"
                    strokeWidth="2"
                    opacity="0.8"
                  />
                  {/* Draw endpoints */}
                  {screenPoints.length > 0 && (
                    <>
                      <circle cx={screenPoints[0].x} cy={screenPoints[0].y} r="3" fill="#00FF00" />
                      <circle cx={screenPoints[screenPoints.length - 1].x} cy={screenPoints[screenPoints.length - 1].y} r="3" fill="#00FF00" />
                    </>
                  )}
                </g>
              )
            } else if (feature.type === 'circle') {
              // Draw circle feature
              const center = feature.parameters.center as [number, number]
              const radius = feature.parameters.radius as number
              const screenCenter = worldToScreen(center[0], center[1])
              return (
                <g key={`circle-${idx}`}>
                  <circle
                    cx={screenCenter.x}
                    cy={screenCenter.y}
                    r={radius * scale}
                    fill="none"
                    stroke="#FF00FF"
                    strokeWidth="2"
                    opacity="0.8"
                  />
                  {/* Draw center point */}
                  <circle cx={screenCenter.x} cy={screenCenter.y} r="3" fill="#FF00FF" />
                </g>
              )
            } else if (feature.type === 'corner') {
              // Draw corner feature
              const screenPoints = feature.points.map(p => worldToScreen(p[0], p[1]))
              const cornerPoint = feature.parameters.corner_point as [number, number]
              const screenCorner = worldToScreen(cornerPoint[0], cornerPoint[1])
              return (
                <g key={`corner-${idx}`}>
                  <polyline
                    points={screenPoints.map(p => `${p.x},${p.y}`).join(' ')}
                    fill="none"
                    stroke="#FFA500"
                    strokeWidth="2"
                    opacity="0.8"
                  />
                  {/* Draw corner point */}
                  <circle cx={screenCorner.x} cy={screenCorner.y} r="4" fill="#FFA500" />
                </g>
              )
            }
            return null
          })}

          {/* Draw robot body (red circle) */}
          <circle
            cx={centerX}
            cy={centerY}
            r={robotRadiusPixels}
            fill="#FF0000"
            stroke="#8B0000"
            strokeWidth="2"
          />

          {/* Draw orientation arrow */}
          <line
            x1={centerX}
            y1={centerY}
            x2={centerX + arrowLength * Math.cos(-robotYawRad)}
            y2={centerY + arrowLength * Math.sin(-robotYawRad)}
            stroke="#000000"
            strokeWidth="3"
          />
        </svg>
        <div style={{ fontSize: '12px', marginTop: '5px' }}>
          Range: {scanData.range_min.toFixed(1)}m - {scanData.range_max.toFixed(1)}m |
          Points: {scanData.ranges.length} |
          Update: {scanData.timestamp}
        </div>
        {objectDetectionData?.features && objectDetectionData.features.length > 0 && (
          <div style={{ fontSize: '10px', marginTop: '3px' }}>
            <span style={{ color: '#00FF00' }}>■ Lines</span>{' '}
            <span style={{ color: '#FF00FF' }}>● Circles</span>{' '}
            <span style={{ color: '#FFA500' }}>▲ Corners</span>
            {' '}({objectDetectionData.features.length} features)
          </div>
        )}
        {robotSpec?.shape?.diameter_m && (
          <div style={{ fontSize: '10px', color: '#FF0000', fontWeight: 'bold', marginTop: '3px' }}>
            🤖 Robot: ⌀ {(robotSpec.shape.diameter_m * 100).toFixed(1)}cm
          </div>
        )}
      </div>
    )
  }

  const getWifiColor = (rssi: number): string => {
    if (rssi >= -50) return '#4CAF50' // Green - Excellent
    if (rssi >= -60) return '#8BC34A' // Light Green - Very Good
    if (rssi >= -70) return '#FFEB3B' // Yellow - Good
    if (rssi >= -80) return '#FF9800' // Orange - Fair
    return '#F44336' // Red - Weak/Very Weak
  }

  const getWifiStrengthText = (rssi: number): string => {
    if (rssi >= -50) return 'Excellent'
    if (rssi >= -60) return 'Very Good'
    if (rssi >= -70) return 'Good'
    if (rssi >= -80) return 'Fair'
    if (rssi >= -90) return 'Weak'
    return 'Very Weak'
  }

  const getWifiBarWidth = (rssi: number): number => {
    // Convert RSSI range (-20 to -90 dBm) to percentage (100% to 0%)
    const minRssi = -90
    const maxRssi = -20
    const percentage = ((rssi - minRssi) / (maxRssi - minRssi)) * 100
    return Math.max(0, Math.min(100, percentage))
  }

  const renderMap = () => {
    if (!mapData || !mapData.data) {
      return <p>No map data available</p>
    }

    const width = Math.min(1200, mapData.info.width * 3)
    const height = Math.min(1200, mapData.info.height * 3)
    const scaleX = width / mapData.info.width
    const scaleY = height / mapData.info.height

    // Create canvas for map visualization
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')

    if (!ctx) return <p>Cannot render map</p>

    // Clear canvas
    ctx.fillStyle = '#f0f0f0'
    ctx.fillRect(0, 0, width, height)

    // Render occupancy grid
    for (let y = 0; y < mapData.info.height; y++) {
      for (let x = 0; x < mapData.info.width; x++) {
        const index = y * mapData.info.width + x
        const value = mapData.data[index]

        let color = '#f0f0f0' // Unknown (gray)
        if (value === 0) {
          color = '#ffffff' // Free space (white)
        } else if (value === 100) {
          color = '#000000' // Occupied (black)
        }

        ctx.fillStyle = color
        const pixelX = Math.floor(x * scaleX)
        const pixelY = Math.floor((mapData.info.height - 1 - y) * scaleY) // Flip Y axis
        ctx.fillRect(pixelX, pixelY, Math.ceil(scaleX), Math.ceil(scaleY))
      }
    }

    // Draw robot pose if available
    if (robotPose) {
      // Convert world coordinates to grid coordinates
      const robotGridX = (robotPose.x - mapData.info.origin.position.x) / mapData.info.resolution
      const robotGridY = (robotPose.y - mapData.info.origin.position.y) / mapData.info.resolution

      // Convert grid coordinates to pixel coordinates
      const robotPixelX = robotGridX * scaleX
      const robotPixelY = (mapData.info.height - 1 - robotGridY) * scaleY // Flip Y axis

      // Get robot diameter from specification, default to 0.35m if not available
      const robotDiameterMeters = robotSpec?.shape?.diameter_m ?? 0.35

      // Convert robot diameter from meters to pixels
      const robotDiameterCells = robotDiameterMeters / mapData.info.resolution
      const robotRadiusPixels = (robotDiameterCells / 2) * scaleX

      // Arrow length should be proportional to robot size
      const arrowLength = robotRadiusPixels * 1.5

      // Convert yaw from degrees to radians (robotPose.yaw is in degrees)
      const yawRad = (robotPose.yaw * Math.PI) / 180

      // Draw robot body (circle)
      ctx.beginPath()
      ctx.arc(robotPixelX, robotPixelY, robotRadiusPixels, 0, 2 * Math.PI)
      ctx.fillStyle = '#FF0000' // Red
      ctx.fill()
      ctx.strokeStyle = '#8B0000' // Dark red border
      ctx.lineWidth = 2
      ctx.stroke()

      // Draw orientation arrow
      const arrowEndX = robotPixelX + arrowLength * Math.cos(-yawRad) // Negative because canvas Y is flipped
      const arrowEndY = robotPixelY + arrowLength * Math.sin(-yawRad) // Negative because canvas Y is flipped

      ctx.beginPath()
      ctx.moveTo(robotPixelX, robotPixelY)
      ctx.lineTo(arrowEndX, arrowEndY)
      ctx.strokeStyle = '#000000' // Black arrow
      ctx.lineWidth = 3
      ctx.stroke()
    }

    return (
      <div style={{ textAlign: 'center' }}>
        <img
          src={canvas.toDataURL()}
          alt="Occupancy Grid Map"
          style={{ border: '1px solid #ccc', maxWidth: '100%' }}
        />
        <div style={{ fontSize: '12px', marginTop: '5px' }}>
          Size: {mapData.info.width}x{mapData.info.height} |
          Resolution: {mapData.info.resolution.toFixed(3)}m/cell |
          Frame: {mapData.header.frame_id}
        </div>
        <div style={{ fontSize: '10px', color: '#666' }}>
          Last updated: {mapData.timestamp}
        </div>
        {robotPose && (
          <div style={{ fontSize: '10px', color: '#FF0000', fontWeight: 'bold', marginTop: '3px' }}>
            🤖 Robot: ({robotPose.x.toFixed(2)}, {robotPose.y.toFixed(2)}) @ {robotPose.yaw.toFixed(1)}°
            {robotSpec?.shape?.diameter_m && (
              <span style={{ marginLeft: '8px', color: '#666' }}>
                | ⌀ {(robotSpec.shape.diameter_m * 100).toFixed(1)}cm
              </span>
            )}
          </div>
        )}
      </div>
    )
  }

  const renderCamera = () => {
    if (!cameraData || !cameraData.data) {
      return <p>No camera data available</p>
    }

    // Convert base64 data back to image
    const imageUrl = `data:image/${cameraData.format === 'jpeg' ? 'jpeg' : 'png'};base64,${cameraData.data}`

    return (
      <div style={{ textAlign: 'center' }}>
        <img
          src={imageUrl}
          alt="Robot Camera Feed"
          style={{
            border: '1px solid #ccc',
            maxWidth: '100%',
            maxHeight: '300px',
            objectFit: 'contain'
          }}
        />
        <div style={{ fontSize: '12px', marginTop: '5px' }}>
          Format: {cameraData.format} |
          Frame: {cameraData.header.frame_id}
        </div>
        <div style={{ fontSize: '10px', color: '#666' }}>
          Last updated: {cameraData.timestamp}
        </div>
      </div>
    )
  }

  const renderNavigation = () => {
    const getStatusColor = (status: string) => {
      switch (status) {
        case 'navigating': return '#2196F3'
        case 'succeeded': return '#4CAF50'
        case 'failed': return '#F44336'
        case 'cancelled': return '#FF9800'
        default: return '#9E9E9E'
      }
    }

    const getStatusIcon = (status: string) => {
      switch (status) {
        case 'navigating': return '🚀'
        case 'succeeded': return '✅'
        case 'failed': return '❌'
        case 'cancelled': return '🛑'
        default: return '⏸️'
      }
    }

    return (
      <div>
        {/* Navigation Status */}
        <div style={{ marginBottom: '15px' }}>
          <h3 style={{ margin: '0 0 10px 0' }}>Status</h3>
          {navigationStatus ? (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              padding: '10px',
              borderRadius: '5px',
              backgroundColor: '#f5f5f5',
              border: `2px solid ${getStatusColor(navigationStatus.status)}`
            }}>
              <span style={{ fontSize: '1.5em', marginRight: '10px' }}>
                {getStatusIcon(navigationStatus.status)}
              </span>
              <div>
                <div style={{
                  fontWeight: 'bold',
                  color: getStatusColor(navigationStatus.status),
                  textTransform: 'capitalize'
                }}>
                  {navigationStatus.status}
                </div>
                <div style={{ fontSize: '0.9em', color: '#666' }}>
                  {navigationStatus.message}
                </div>
                <div style={{ fontSize: '0.8em', color: '#999' }}>
                  {navigationStatus.timestamp}
                </div>
              </div>
            </div>
          ) : (
            <div style={{ padding: '10px', backgroundColor: '#f5f5f5', borderRadius: '5px' }}>
              <span>⏸️ No navigation status</span>
            </div>
          )}
        </div>

        {/* Navigation Feedback */}
        {navigationFeedback && navigationStatus?.status === 'navigating' && (
          <div style={{ marginBottom: '15px' }}>
            <h3 style={{ margin: '0 0 10px 0' }}>Progress</h3>
            <div style={{
              padding: '10px',
              backgroundColor: '#e3f2fd',
              borderRadius: '5px',
              border: '1px solid #2196F3'
            }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                <div>
                  <div style={{ fontSize: '0.9em', color: '#666' }}>Distance Remaining</div>
                  <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#2196F3' }}>
                    {navigationFeedback.distance_remaining.toFixed(2)}m
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.9em', color: '#666' }}>ETA</div>
                  <div style={{ fontSize: '1.2em', fontWeight: 'bold', color: '#2196F3' }}>
                    {navigationFeedback.estimated_time_remaining}s
                  </div>
                </div>
              </div>
              <div style={{ fontSize: '0.8em', color: '#666' }}>
                Current Pose: ({navigationFeedback.current_pose.x.toFixed(2)}, {navigationFeedback.current_pose.y.toFixed(2)})
              </div>
              <div style={{ fontSize: '0.8em', color: '#999' }}>
                {navigationFeedback.timestamp}
              </div>
            </div>
          </div>
        )}

        {/* Navigation Controls */}
        <div>
          <h3 style={{ margin: '0 0 10px 0' }}>Send Goal</h3>

          {/* Navigation Mode Toggle */}
          <div style={{ marginBottom: '15px' }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.9em' }}>
              <input
                type="checkbox"
                checked={navGoalRelative}
                onChange={(e) => setNavGoalRelative(e.target.checked)}
              />
              <span style={{ fontWeight: navGoalRelative ? 'bold' : 'normal', color: navGoalRelative ? '#2196F3' : '#333' }}>
                Relative Navigation {navGoalRelative ? '(relative to robot)' : '(absolute coordinates)'}
              </span>
            </label>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', marginBottom: '10px' }}>
            <div>
              <label style={{ fontSize: '0.9em', display: 'block' }}>
                {navGoalRelative ? 'Forward/Backward (m):' : 'X (m):'}
              </label>
              <input
                type="number"
                step="0.1"
                value={navGoalX}
                onChange={(e) => setNavGoalX(Number(e.target.value))}
                style={{ width: '100%' }}
                placeholder={navGoalRelative ? "Positive = forward" : "Map X coordinate"}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.9em', display: 'block' }}>
                {navGoalRelative ? 'Left/Right (m):' : 'Y (m):'}
              </label>
              <input
                type="number"
                step="0.1"
                value={navGoalY}
                onChange={(e) => setNavGoalY(Number(e.target.value))}
                style={{ width: '100%' }}
                placeholder={navGoalRelative ? "Positive = left" : "Map Y coordinate"}
              />
            </div>
            <div>
              <label style={{ fontSize: '0.9em', display: 'block' }}>
                {navGoalRelative ? 'Turn (deg):' : 'Yaw (deg):'}
              </label>
              <input
                type="number"
                step="5"
                value={navGoalYaw}
                onChange={(e) => setNavGoalYaw(Number(e.target.value))}
                style={{ width: '100%' }}
                placeholder={navGoalRelative ? "Relative rotation" : "Absolute orientation"}
              />
            </div>
          </div>
          <div style={{ display: 'flex', gap: '10px' }}>
            <button
              onClick={sendNavigationGoal}
              disabled={!robotConnected}
              style={{
                flex: 1,
                backgroundColor: navGoalRelative ? '#2196F3' : '#4CAF50',
                color: 'white',
                border: 'none',
                padding: '10px',
                borderRadius: '5px'
              }}
            >
              {navGoalRelative ? '🧭 Navigate Relative' : '🎯 Navigate to Pose'}
            </button>
            <button
              onClick={cancelNavigation}
              disabled={!robotConnected || navigationStatus?.status !== 'navigating'}
              style={{
                backgroundColor: '#F44336',
                color: 'white',
                border: 'none',
                padding: '10px 15px',
                borderRadius: '5px'
              }}
            >
              🛑 Cancel
            </button>
          </div>
        </div>
      </div>
    )
  }

  const renderRobotPose = () => {
    if (!robotPose) {
      return <p>No robot pose data available</p>
    }

    return (
      <div>
        <div style={{ marginBottom: '15px' }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#2196F3' }}>🤖 Current Position (TF2)</h3>
          <div style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: '15px',
            marginBottom: '15px'
          }}>
            <div style={{
              padding: '12px',
              backgroundColor: '#e3f2fd',
              borderRadius: '8px',
              border: '1px solid #2196F3'
            }}>
              <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>Position (m)</div>
              <div style={{ fontSize: '1.3em', fontWeight: 'bold', color: '#1976D2' }}>
                X: {robotPose.x.toFixed(3)}
              </div>
              <div style={{ fontSize: '1.3em', fontWeight: 'bold', color: '#1976D2' }}>
                Y: {robotPose.y.toFixed(3)}
              </div>
              <div style={{ fontSize: '0.8em', color: '#666', marginTop: '5px' }}>
                Via TF2 Transform
              </div>
            </div>

            <div style={{
              padding: '12px',
              backgroundColor: '#fff3e0',
              borderRadius: '8px',
              border: '1px solid #FF9800'
            }}>
              <div style={{ fontSize: '0.9em', color: '#666', marginBottom: '5px' }}>Orientation</div>
              <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#F57C00' }}>
                {robotPose.yaw.toFixed(1)}°
              </div>
              <div style={{ fontSize: '0.8em', color: '#666', marginTop: '5px' }}>
                Yaw angle
              </div>
            </div>
          </div>

          <div style={{
            padding: '10px',
            backgroundColor: '#f5f5f5',
            borderRadius: '5px',
            fontSize: '0.85em'
          }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
              <div>
                <strong>Frame:</strong> {robotPose.frame_id}
              </div>
              <div>
                <strong>Updated:</strong> {robotPose.timestamp}
              </div>
            </div>
            <div style={{ marginTop: '8px', fontSize: '0.8em', color: '#666' }}>
              <strong>Source:</strong> TF2 transform from {robotPose.frame_id} to base_footprint
            </div>
          </div>
        </div>
      </div>
    )
  }


  const renderObjectDetection = () => {
    if (!objectDetectionData) {
      return <p>Loading object detection data...</p>
    }

    const { objects, summary, features } = objectDetectionData

    if (!summary || !objects) {
      return <p>No object detection data available</p>
    }

    const getObjectIcon = (type: string) => {
      const icons: { [key: string]: string } = {
        'chair': '🪑',
        'table': '🪴',
        'wall': '🧱',
        'person': '🚶',
        'box': '📦',
        'plant': '🌱',
        'door': '🚪'
      }
      return icons[type] || '📍'
    }

    const getConfidenceColor = (confidence: number) => {
      if (confidence >= 0.8) return '#4CAF50'  // Green - High confidence
      if (confidence >= 0.6) return '#FF9800'  // Orange - Medium confidence
      return '#F44336'  // Red - Low confidence
    }

    const getFeatureIcon = (type: string) => {
      const icons: { [key: string]: string } = {
        'line': '━',
        'circle': '●',
        'corner': '▲'
      }
      return icons[type] || '■'
    }

    const getFeatureColor = (type: string) => {
      const colors: { [key: string]: string } = {
        'line': '#00FF00',
        'circle': '#FF00FF',
        'corner': '#FFA500'
      }
      return colors[type] || '#2196F3'
    }

    return (
      <div>
        {/* Summary */}
        <div style={{
          backgroundColor: '#e3f2fd',
          padding: '15px',
          borderRadius: '8px',
          marginBottom: '15px',
          border: '1px solid #2196F3'
        }}>
          <h3 style={{ margin: '0 0 10px 0', color: '#1976D2' }}>
            🔍 Detection Summary
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
            <div>
              <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#2196F3' }}>
                {summary.total_objects}
              </div>
              <div style={{ fontSize: '0.9em', color: '#666' }}>Total Objects</div>
            </div>
            <div>
              <div style={{ fontSize: '0.9em', color: '#666' }}>Object Types:</div>
              {summary.object_counts ? Object.entries(summary.object_counts).map(([type, count]) => (
                <div key={type} style={{ fontSize: '0.8em' }}>
                  {getObjectIcon(type)} {type}: {count}
                </div>
              )) : <div style={{ fontSize: '0.8em', color: '#999' }}>No objects detected</div>}
            </div>
          </div>
          <div style={{ fontSize: '0.8em', color: '#999', marginTop: '10px' }}>
            Last update: {summary.last_update || 'Never'} |
            Scan available: {summary.scan_available ? 'Yes' : 'No'}
          </div>
        </div>

        {/* Feature Details */}
        {features && features.length > 0 && (
          <div style={{
            backgroundColor: '#f0f0f0',
            padding: '10px',
            borderRadius: '5px',
            marginBottom: '15px'
          }}>
            <h4 style={{ margin: '0 0 8px 0', fontSize: '0.9em', color: '#666' }}>
              Detected Features ({features.length})
            </h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {features.map((feature, idx) => (
                <div
                  key={idx}
                  style={{
                    backgroundColor: 'white',
                    border: `2px solid ${getFeatureColor(feature.type)}`,
                    borderRadius: '4px',
                    padding: '4px 8px',
                    fontSize: '0.75em',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}
                >
                  <span style={{ color: getFeatureColor(feature.type), fontWeight: 'bold' }}>
                    {getFeatureIcon(feature.type)}
                  </span>
                  <span style={{ textTransform: 'capitalize' }}>{feature.type}</span>
                  <span style={{ color: '#999' }}>({feature.points.length} pts)</span>
                  <span style={{
                    backgroundColor: getConfidenceColor(feature.confidence),
                    color: 'white',
                    padding: '1px 4px',
                    borderRadius: '3px',
                    marginLeft: '4px'
                  }}>
                    {(feature.confidence * 100).toFixed(0)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Object List */}
        {objects.length > 0 ? (
          <div style={{
            maxHeight: '400px',
            overflowY: 'auto',
            border: '1px solid #ddd',
            borderRadius: '5px'
          }}>
            {objects.map((obj, index) => (
              <div
                key={index}
                style={{
                  padding: '10px',
                  borderBottom: index < objects.length - 1 ? '1px solid #eee' : 'none',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                  <span style={{ fontSize: '1.5em' }}>{getObjectIcon(obj.type)}</span>
                  <div>
                    <div style={{ fontWeight: 'bold', textTransform: 'capitalize' }}>
                      {obj.type}
                    </div>
                    <div style={{ fontSize: '0.8em', color: '#666' }}>
                      {obj.distance}m @ {obj.angle_deg}° ({obj.x.toFixed(1)}, {obj.y.toFixed(1)})
                    </div>
                  </div>
                </div>
                <div style={{
                  backgroundColor: getConfidenceColor(obj.confidence),
                  color: 'white',
                  padding: '4px 8px',
                  borderRadius: '12px',
                  fontSize: '0.8em',
                  fontWeight: 'bold'
                }}>
                  {(obj.confidence * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div style={{
            textAlign: 'center',
            padding: '30px',
            color: '#666',
            backgroundColor: '#f5f5f5',
            borderRadius: '5px'
          }}>
            🔍 No objects detected
          </div>
        )}
      </div>
    )
  }

  const renderConsole = () => {
    const consoleRef = React.useRef<HTMLDivElement>(null)

    // Auto-scroll to bottom when new messages arrive
    React.useEffect(() => {
      if (consoleRef.current) {
        consoleRef.current.scrollTop = consoleRef.current.scrollHeight
      }
    }, [consoleMessages])

    return (
      <div>
        <div
          ref={consoleRef}
          style={{
            height: '300px',
            overflowY: 'auto',
            backgroundColor: '#1a1a1a',
            color: '#00ff00',
            fontFamily: 'monospace',
            fontSize: '0.9em',
            padding: '10px',
            border: '2px solid #333',
            borderRadius: '5px'
          }}
        >
          {consoleMessages.length === 0 ? (
            <div style={{ color: '#666', fontStyle: 'italic' }}>
              Waiting for robot messages...
            </div>
          ) : (
            consoleMessages.map((msg, index) => (
              <div key={index} style={{
                marginBottom: '5px',
                borderBottom: '1px solid #333',
                paddingBottom: '3px'
              }}>
                <span style={{ color: '#888', fontSize: '0.8em' }}>
                  [{new Date(msg.timestamp * 1000).toLocaleTimeString()}]
                </span>
                <span style={{ marginLeft: '8px' }}>
                  {msg.content}
                </span>
              </div>
            ))
          )}
        </div>
        <div style={{
          fontSize: '0.8em',
          color: '#666',
          marginTop: '5px',
          textAlign: 'center'
        }}>
          Console output from robot application • {consoleMessages.length} messages
        </div>
      </div>
    )
  }

  const renderGeneratedGrid = () => {
    if (!generatedGrid) {
      return <p>No grid generated yet</p>
    }

    const width = Math.min(300, gridWidth)
    const height = Math.min(300, gridHeight)
    const scaleX = width / gridWidth
    const scaleY = height / gridHeight

    // Create canvas for grid visualization
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    const ctx = canvas.getContext('2d')

    if (!ctx) return <p>Cannot render grid</p>

    // Clear canvas
    ctx.fillStyle = '#f0f0f0'
    ctx.fillRect(0, 0, width, height)

    // Render occupancy grid
    for (let y = 0; y < gridHeight; y++) {
      for (let x = 0; x < gridWidth; x++) {
        const index = y * gridWidth + x
        const value = generatedGrid[index]

        let color = '#c0c0c0' // Unknown (light gray)
        if (value === 0) {
          color = '#ffffff' // Free space (white)
        } else if (value === 100) {
          color = '#000000' // Occupied (black)
        }

        ctx.fillStyle = color
        const pixelX = Math.floor(x * scaleX)
        const pixelY = Math.floor((gridHeight - 1 - y) * scaleY) // Flip Y axis
        ctx.fillRect(pixelX, pixelY, Math.ceil(scaleX), Math.ceil(scaleY))
      }
    }

    return (
      <div style={{ textAlign: 'center' }}>
        <img
          src={canvas.toDataURL()}
          alt="Generated Occupancy Grid"
          style={{ border: '1px solid #ccc', maxWidth: '100%' }}
        />
        <div style={{ fontSize: '12px', marginTop: '5px' }}>
          Size: {gridWidth}x{gridHeight} |
          Resolution: {gridResolution.toFixed(3)}m/cell |
          Preview of generated grid
        </div>
      </div>
    )
  }

  return (
    <div className="container" style={{ position: 'relative' }}>
      <h1>Robot Control Dashboard</h1>

      <div className="status">
        <span>Robot Status:</span>
        <div className={`status-indicator ${robotConnected ? 'connected' : 'disconnected'}`}></div>
        <span>{robotConnected ? 'Connected' : 'Disconnected'}</span>
      </div>

      {connectionError && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(128, 128, 128, 0.9)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
          pointerEvents: 'all'
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
            textAlign: 'center',
            maxWidth: '600px'
          }}>
            <div style={{ fontSize: '2em', marginBottom: '20px' }}>❌</div>
            <div style={{ fontSize: '1.5em', fontWeight: 'bold', color: '#F44336', marginBottom: '10px' }}>
              Connection Error
            </div>
            <div style={{ fontSize: '1em', color: '#666' }}>
              {connectionError}
            </div>
            <div style={{ fontSize: '0.9em', color: '#999', marginTop: '20px' }}>
              <strong>Expected URL parameters:</strong><br/>
              session_id, robot_id, token, api_url
            </div>
          </div>
        </div>
      )}

      {!robotConnected && !connectionError && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: 'rgba(128, 128, 128, 0.7)',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          zIndex: 1000,
          pointerEvents: 'all'
        }}>
          <div style={{
            backgroundColor: 'white',
            padding: '30px',
            borderRadius: '10px',
            boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)',
            textAlign: 'center',
            fontSize: '1.5em',
            fontWeight: 'bold',
            color: '#333'
          }}>
            🔌 Waiting for robot to connect...
          </div>
        </div>
      )}

      <div className="grid" style={{
        filter: !robotConnected ? 'grayscale(100%) brightness(0.8)' : 'none',
        pointerEvents: !robotConnected ? 'none' : 'auto'
      }}>
        <div className="panel">
          <h2>Controls</h2>
          <div className="controls">

            <div className="control-group">
              <label>Velocity Control (Twist):</label>
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginTop: '5px' }}>
                <div>
                  <label style={{ fontSize: '0.9em', display: 'block' }}>Linear X (m/s):</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="0.0"
                    value={linearX}
                    onChange={(e) => setLinearX(Number(e.target.value))}
                    style={{ width: '80px' }}
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.9em', display: 'block' }}>Angular Z (rad/s):</label>
                  <input
                    type="number"
                    step="0.1"
                    placeholder="0.0"
                    value={angularZ}
                    onChange={(e) => setAngularZ(Number(e.target.value))}
                    style={{ width: '80px' }}
                  />
                </div>
                <button onClick={() => sendTwistCommand()} disabled={!robotConnected} style={{ marginTop: '20px' }}>
                  Send Twist
                </button>
              </div>
            </div>

            <div className="control-group">
              <label>Quick Twist Commands (click multiple times to accelerate):</label>
              <div style={{ display: 'flex', gap: '5px', flexWrap: 'wrap', marginTop: '5px' }}>
                <button
                  onClick={accelerateForward}
                  disabled={!robotConnected}
                  style={{
                    backgroundColor: currentLinearSpeed > 0 ? `rgb(${Math.min(255, 100 + currentLinearSpeed * 155)}, ${Math.max(0, 200 - currentLinearSpeed * 100)}, 0)` : undefined
                  }}
                >
                  Forward {currentLinearSpeed > 0 && `(${currentLinearSpeed.toFixed(1)})`}
                </button>
                <button
                  onClick={accelerateBackward}
                  disabled={!robotConnected}
                  style={{
                    backgroundColor: currentLinearSpeed < 0 ? `rgb(${Math.min(255, 100 + Math.abs(currentLinearSpeed) * 155)}, ${Math.max(0, 200 - Math.abs(currentLinearSpeed) * 100)}, 0)` : undefined
                  }}
                >
                  Backward {currentLinearSpeed < 0 && `(${Math.abs(currentLinearSpeed).toFixed(1)})`}
                </button>
                <button
                  onClick={accelerateRotateRight}
                  disabled={!robotConnected}
                  style={{
                    backgroundColor: currentAngularSpeed > 0 ? `rgb(0, ${Math.max(0, 200 - currentAngularSpeed * 50)}, ${Math.min(255, 100 + currentAngularSpeed * 77)})` : undefined
                  }}
                >
                  Rotate Right {currentAngularSpeed > 0 && `(${currentAngularSpeed.toFixed(1)})`}
                </button>
                <button
                  onClick={accelerateRotateLeft}
                  disabled={!robotConnected}
                  style={{
                    backgroundColor: currentAngularSpeed < 0 ? `rgb(0, ${Math.max(0, 200 - Math.abs(currentAngularSpeed) * 50)}, ${Math.min(255, 100 + Math.abs(currentAngularSpeed) * 77)})` : undefined
                  }}
                >
                  Rotate Left {currentAngularSpeed < 0 && `(${Math.abs(currentAngularSpeed).toFixed(1)})`}
                </button>
                <button
                  onClick={stopRobot}
                  disabled={!robotConnected}
                  style={{ backgroundColor: '#F44336', color: 'white' }}
                >
                  Stop
                </button>
              </div>
              <div style={{ fontSize: '0.8em', color: '#666', marginTop: '8px' }}>
                Speed: Linear {Math.abs(currentLinearSpeed).toFixed(1)} m/s (max {MAX_LINEAR_SPEED}) |
                Angular {Math.abs(currentAngularSpeed).toFixed(1)} rad/s (max {MAX_ANGULAR_SPEED})
              </div>
            </div>

            <div className="control-group">
              <label>Test Functions:</label>
              <div style={{ display: 'flex', justifyContent: 'center', gap: '15px', marginTop: '10px' }}>
                <button
                  onClick={sendTestCommand}
                  disabled={!robotConnected}
                  style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    backgroundColor: robotConnected ? '#4CAF50' : '#ccc',
                    color: 'white',
                    border: 'none',
                    fontSize: '1.2em',
                    fontWeight: 'bold',
                    cursor: robotConnected ? 'pointer' : 'not-allowed',
                    transition: 'all 0.3s ease',
                    boxShadow: robotConnected ? '0 4px 8px rgba(76, 175, 80, 0.3)' : 'none'
                  }}
                  onMouseOver={(e) => {
                    if (robotConnected) {
                      e.currentTarget.style.backgroundColor = '#45a049'
                      e.currentTarget.style.transform = 'scale(1.05)'
                    }
                  }}
                  onMouseOut={(e) => {
                    if (robotConnected) {
                      e.currentTarget.style.backgroundColor = '#4CAF50'
                      e.currentTarget.style.transform = 'scale(1)'
                    }
                  }}
                >
                  🧪 Test
                </button>
                <button
                  onClick={sendTestWSCommand}
                  disabled={!socket || !socket.connected}
                  style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    backgroundColor: (socket && socket.connected) ? '#2196F3' : '#ccc',
                    color: 'white',
                    border: 'none',
                    fontSize: '1em',
                    fontWeight: 'bold',
                    cursor: (socket && socket.connected) ? 'pointer' : 'not-allowed',
                    transition: 'all 0.3s ease',
                    boxShadow: (socket && socket.connected) ? '0 4px 8px rgba(33, 150, 243, 0.3)' : 'none'
                  }}
                  onMouseOver={(e) => {
                    if (socket && socket.connected) {
                      e.currentTarget.style.backgroundColor = '#1976D2'
                      e.currentTarget.style.transform = 'scale(1.05)'
                    }
                  }}
                  onMouseOut={(e) => {
                    if (socket && socket.connected) {
                      e.currentTarget.style.backgroundColor = '#2196F3'
                      e.currentTarget.style.transform = 'scale(1)'
                    }
                  }}
                >
                  📡 Test WS
                </button>
                <button
                  onClick={stopApp}
                  disabled={!robotConnected}
                  style={{
                    width: '80px',
                    height: '80px',
                    borderRadius: '50%',
                    backgroundColor: robotConnected ? '#F44336' : '#ccc',
                    color: 'white',
                    border: 'none',
                    fontSize: '1em',
                    fontWeight: 'bold',
                    cursor: robotConnected ? 'pointer' : 'not-allowed',
                    transition: 'all 0.3s ease',
                    boxShadow: robotConnected ? '0 4px 8px rgba(244, 67, 54, 0.3)' : 'none'
                  }}
                  onMouseOver={(e) => {
                    if (robotConnected) {
                      e.currentTarget.style.backgroundColor = '#d32f2f'
                      e.currentTarget.style.transform = 'scale(1.05)'
                    }
                  }}
                  onMouseOut={(e) => {
                    if (robotConnected) {
                      e.currentTarget.style.backgroundColor = '#F44336'
                      e.currentTarget.style.transform = 'scale(1)'
                    }
                  }}
                >
                  🛑 Stop
                </button>
              </div>
            </div>

          </div>
        </div>

        <div className="panel">
          <h2>Battery Status</h2>
          <div className="battery-display">
            {batteryData ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '2.5em',
                  fontWeight: 'bold',
                  color: batteryData.level > 50 ? '#4CAF50' : batteryData.level > 20 ? '#FF9800' : '#F44336'
                }}>
                  {batteryData.level.toFixed(1)}%
                </div>
                <div style={{
                  width: '200px',
                  height: '30px',
                  border: '2px solid #333',
                  borderRadius: '5px',
                  margin: '10px auto',
                  position: 'relative',
                  backgroundColor: '#f0f0f0'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${Math.max(0, Math.min(100, batteryData.level))}%`,
                    backgroundColor: batteryData.level > 50 ? '#4CAF50' : batteryData.level > 20 ? '#FF9800' : '#F44336',
                    borderRadius: '3px',
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
                <div style={{ fontSize: '0.9em', color: '#666' }}>
                  Last updated: {batteryData.timestamp}
                </div>
              </div>
            ) : (
              <p>No battery data available</p>
            )}
          </div>
        </div>

        <div className="panel">
          <h2>WiFi Status</h2>
          <div className="wifi-display">
            {wifiData ? (
              <div style={{ textAlign: 'center' }}>
                <div style={{
                  fontSize: '2em',
                  fontWeight: 'bold',
                  color: getWifiColor(wifiData.rssi_dbm)
                }}>
                  {wifiData.rssi_dbm.toFixed(1)} dBm
                </div>
                <div style={{
                  fontSize: '1.2em',
                  margin: '10px 0',
                  color: getWifiColor(wifiData.rssi_dbm),
                  textTransform: 'capitalize'
                }}>
                  {getWifiStrengthText(wifiData.rssi_dbm)}
                </div>
                <div style={{
                  width: '180px',
                  height: '20px',
                  border: '2px solid #333',
                  borderRadius: '10px',
                  margin: '10px auto',
                  position: 'relative',
                  backgroundColor: '#f0f0f0'
                }}>
                  <div style={{
                    height: '100%',
                    width: `${getWifiBarWidth(wifiData.rssi_dbm)}%`,
                    backgroundColor: getWifiColor(wifiData.rssi_dbm),
                    borderRadius: '8px',
                    transition: 'width 0.3s ease'
                  }}></div>
                </div>
                <div style={{ fontSize: '0.9em', color: '#666' }}>
                  Signal Strength
                </div>
                <div style={{ fontSize: '0.8em', color: '#999', marginTop: '5px' }}>
                  Last updated: {wifiData.timestamp}
                </div>
              </div>
            ) : (
              <p>No WiFi data available</p>
            )}
          </div>
        </div>

        <div className="panel">
          <h2>Velocity Monitor (/cmd_vel)</h2>
          <div className="cmd-vel-display">
            {cmdVelData && cmdVelData.status !== 'no_data' ? (
              <div>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '10px' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.1em', fontWeight: 'bold', marginBottom: '5px', color: '#2196F3' }}>
                      Linear Velocity
                    </div>
                    <div style={{ fontSize: '1.4em', fontWeight: 'bold', color: Math.abs(cmdVelData.linear.x) > 0.01 ? '#4CAF50' : '#666' }}>
                      X: {cmdVelData.linear.x.toFixed(3)} m/s
                    </div>
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      Y: {cmdVelData.linear.y.toFixed(3)} m/s
                    </div>
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      Z: {cmdVelData.linear.z.toFixed(3)} m/s
                    </div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '1.1em', fontWeight: 'bold', marginBottom: '5px', color: '#FF9800' }}>
                      Angular Velocity
                    </div>
                    <div style={{ fontSize: '1.4em', fontWeight: 'bold', color: Math.abs(cmdVelData.angular.z) > 0.01 ? '#4CAF50' : '#666' }}>
                      Z: {cmdVelData.angular.z.toFixed(3)} rad/s
                    </div>
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      X: {cmdVelData.angular.x.toFixed(3)} rad/s
                    </div>
                    <div style={{ fontSize: '0.9em', color: '#666' }}>
                      Y: {cmdVelData.angular.y.toFixed(3)} rad/s
                    </div>
                  </div>
                </div>
                <div style={{ textAlign: 'center', fontSize: '0.8em', color: '#999', borderTop: '1px solid #eee', paddingTop: '8px' }}>
                  Last updated: {cmdVelData.timestamp}
                </div>
                <div style={{ textAlign: 'center', marginTop: '8px' }}>
                  <div style={{
                    display: 'inline-block',
                    padding: '4px 8px',
                    borderRadius: '12px',
                    backgroundColor: (Math.abs(cmdVelData.linear.x) > 0.01 || Math.abs(cmdVelData.angular.z) > 0.01) ? '#4CAF50' : '#9E9E9E',
                    color: 'white',
                    fontSize: '0.8em',
                    fontWeight: 'bold'
                  }}>
                    {(Math.abs(cmdVelData.linear.x) > 0.01 || Math.abs(cmdVelData.angular.z) > 0.01) ? 'MOVING' : 'STOPPED'}
                  </div>
                </div>
              </div>
            ) : (
              <p>No velocity commands received</p>
            )}
          </div>
        </div>

<div className="panel">
          <h2>Laser Scan</h2>
          <div className="laser-scan">
            {renderLaserScan()}
          </div>
        </div>

        <div className="panel">
          <h2>Camera Feed</h2>
          <div className="camera-display">
            {renderCamera()}
          </div>
        </div>

        <div className="panel">
          <h2>Navigation</h2>
          <div className="navigation-display">
            {renderNavigation()}
          </div>
        </div>

        <div className="panel">
          <h2>Robot Pose</h2>
          <div className="pose-display">
            {renderRobotPose()}
          </div>
        </div>

        <div className="panel">
          <h2>Map (/map)</h2>
          <div className="map-display">
            {renderMap()}
          </div>
        </div>

        <div className="panel">
          <h2>OccupancyGrid Generator</h2>
          <div className="occupancy-grid-controls">
            <div className="control-group">
              <label>Grid Parameters:</label>
              <div style={{ display: 'flex', gap: '10px', marginTop: '5px', flexWrap: 'wrap' }}>
                <div>
                  <label style={{ fontSize: '0.9em', display: 'block' }}>Width:</label>
                  <input
                    type="number"
                    value={gridWidth}
                    onChange={(e) => setGridWidth(Number(e.target.value))}
                    style={{ width: '80px' }}
                    min="10"
                    max="1000"
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.9em', display: 'block' }}>Height:</label>
                  <input
                    type="number"
                    value={gridHeight}
                    onChange={(e) => setGridHeight(Number(e.target.value))}
                    style={{ width: '80px' }}
                    min="10"
                    max="1000"
                  />
                </div>
                <div>
                  <label style={{ fontSize: '0.9em', display: 'block' }}>Resolution (m/cell):</label>
                  <input
                    type="number"
                    step="0.01"
                    value={gridResolution}
                    onChange={(e) => setGridResolution(Number(e.target.value))}
                    style={{ width: '80px' }}
                    min="0.01"
                    max="1.0"
                  />
                </div>
              </div>
            </div>

            <div className="control-group">
              <button onClick={generateOccupancyGrid} style={{ marginRight: '10px' }}>
                Generate Fake Grid
              </button>
              <button
                onClick={sendOccupancyGrid}
                disabled={!robotConnected || !generatedGrid}
                style={{ backgroundColor: generatedGrid ? '#4CAF50' : '#ccc' }}
              >
                Send to Robot
              </button>
            </div>

            <div style={{ marginTop: '15px' }}>
              {renderGeneratedGrid()}
            </div>
          </div>
        </div>

        <div className="panel">
          <h2>Object Detection</h2>
          <div className="object-detection">
            {renderObjectDetection()}
          </div>
        </div>


        <div className="panel">
          <h2>Console</h2>
          <div className="console-display">
            {renderConsole()}
          </div>
        </div>

        <div className="panel">
          <h2>System Info</h2>
          <div className="sensor-data">
            Socket: {(socket && socket.connected) ? 'Connected' : 'Disconnected'}<br/>
            Robot: {robotConnected ? 'Online' : 'Offline'}<br/>
            Last Update: {lastDataUpdate || 'Never'}<br/>
            Session ID: {sessionId || 'N/A'}<br/>
            Robot ID: {robotId || 'N/A'}<br/>
            API URL: {apiUrl}
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
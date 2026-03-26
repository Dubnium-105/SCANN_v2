function isFiniteNumber(value) {
  return Number.isFinite(Number(value))
}

function sampledFiniteValues(pixels, maxSamples = 200000) {
  if (!pixels || pixels.length === 0) {
    return []
  }

  const values = []
  const step = Math.max(1, Math.floor(pixels.length / maxSamples))
  for (let index = 0; index < pixels.length; index += step) {
    const value = Number(pixels[index])
    if (Number.isFinite(value)) {
      values.push(value)
    }
  }
  return values
}

function percentileFromSorted(sortedValues, percentile) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
    return 0
  }
  const rank = ((percentile / 100) * (sortedValues.length - 1))
  const lowerIndex = Math.floor(rank)
  const upperIndex = Math.ceil(rank)
  if (lowerIndex === upperIndex) {
    return sortedValues[lowerIndex]
  }
  const fraction = rank - lowerIndex
  return sortedValues[lowerIndex] + (sortedValues[upperIndex] - sortedValues[lowerIndex]) * fraction
}

function medianFromSorted(sortedValues) {
  return percentileFromSorted(sortedValues, 50)
}

function standardDeviation(values) {
  if (!Array.isArray(values) || values.length === 0) {
    return 0
  }
  let sum = 0
  for (const value of values) {
    sum += value
  }
  const mean = sum / values.length
  let variance = 0
  for (const value of values) {
    const delta = value - mean
    variance += delta * delta
  }
  return Math.sqrt(variance / values.length)
}

function sigmaClippedStats(values, sigma = 3, maxIterations = 5) {
  let working = Array.isArray(values) ? [...values] : []
  if (working.length === 0) {
    return { median: 0, std: 0 }
  }

  for (let iteration = 0; iteration < maxIterations; iteration += 1) {
    const sorted = [...working].sort((a, b) => a - b)
    const median = medianFromSorted(sorted)
    const std = standardDeviation(working)
    if (!Number.isFinite(std) || std <= 0) {
      return { median, std: 0 }
    }

    const next = working.filter((value) => Math.abs(value - median) <= sigma * std)
    if (next.length === 0 || next.length === working.length) {
      return { median, std }
    }
    working = next
  }

  const sorted = [...working].sort((a, b) => a - b)
  return {
    median: medianFromSorted(sorted),
    std: standardDeviation(working),
  }
}

function ensureValidInterval(low, high, sortedValues) {
  if (!Array.isArray(sortedValues) || sortedValues.length === 0) {
    return { min: 0, max: 1 }
  }
  const dataMin = sortedValues[0]
  const dataMax = sortedValues[sortedValues.length - 1]

  let safeLow = Number.isFinite(low) ? low : dataMin
  let safeHigh = Number.isFinite(high) ? high : dataMax
  if (safeHigh <= safeLow) {
    if (dataMax > dataMin) {
      safeLow = dataMin
      safeHigh = dataMax
    } else {
      safeLow = dataMin
      safeHigh = dataMin + 1
    }
  }
  return { min: safeLow, max: safeHigh }
}

function adaptiveHighPercentileFromTailRatio(sortedValues) {
  const baseValue = percentileFromSorted(sortedValues, 99.5)
  const probeValue = percentileFromSorted(sortedValues, 99.9)
  const tailRatio = baseValue > 0 ? probeValue / baseValue : 1

  const effective = (-0.25261547 * (tailRatio ** 2)) + (0.53000827 * tailRatio) + 99.48221607
  return {
    effectiveHighPercentile: Math.min(99.8, Math.max(99.2, effective)),
    tailRatio,
  }
}

function brightnessMatchAnchors(pixels, options = {}) {
  const {
    maxSamples = 200000,
    highPercentile = 99.9,
    highlightSigma = 5.0,
    adaptiveHighPercentile = false,
  } = options

  const values = sampledFiniteValues(pixels, maxSamples)
  if (values.length === 0) {
    throw new Error('image contains no finite pixels')
  }

  const sortedValues = [...values].sort((a, b) => a - b)
  const { median, std } = sigmaClippedStats(values)
  let effectiveHighPercentile = highPercentile
  let tailRatio = null
  if (adaptiveHighPercentile) {
    const adaptive = adaptiveHighPercentileFromTailRatio(sortedValues)
    effectiveHighPercentile = adaptive.effectiveHighPercentile
    tailRatio = adaptive.tailRatio
  }

  const percentileHigh = percentileFromSorted(sortedValues, effectiveHighPercentile)
  const sigmaHigh = Number.isFinite(std) && std > 0 ? median + highlightSigma * std : percentileHigh
  return {
    sortedValues,
    backgroundAnchor: median,
    highlightAnchor: Math.max(percentileHigh, sigmaHigh),
    median,
    std,
    highPercentileValue: percentileHigh,
    effectiveHighPercentile,
    tailRatio,
  }
}

export function computeBrightnessMatchIntervalFromPixels(pixels, options = {}) {
  const {
    backgroundPosition = 0.10,
    highlightPosition = 0.98,
  } = options

  if (!(highlightPosition > backgroundPosition)) {
    throw new Error('highlightPosition must be greater than backgroundPosition')
  }

  const anchors = brightnessMatchAnchors(pixels, options)
  const displayMin = (
    highlightPosition * anchors.backgroundAnchor
    - backgroundPosition * anchors.highlightAnchor
  ) / (highlightPosition - backgroundPosition)
  const displayMax = displayMin + (
    (anchors.highlightAnchor - anchors.backgroundAnchor)
    / (highlightPosition - backgroundPosition)
  )
  const interval = ensureValidInterval(displayMin, displayMax, anchors.sortedValues)

  return {
    ...anchors,
    displayMin: interval.min,
    displayMax: interval.max,
    backgroundPosition,
    highlightPosition,
  }
}

export function inferMatchPositionsFromTargetInterval(pixels, targetMin, targetMax, options = {}) {
  if (!(targetMax > targetMin)) {
    throw new Error('targetMax must be greater than targetMin')
  }

  const anchors = brightnessMatchAnchors(pixels, options)
  const width = targetMax - targetMin
  return {
    ...anchors,
    displayMin: targetMin,
    displayMax: targetMax,
    backgroundPosition: (anchors.backgroundAnchor - targetMin) / width,
    highlightPosition: (anchors.highlightAnchor - targetMin) / width,
  }
}

export function buildViewStretchState(node, stretchMin, stretchMax) {
  const pixels = node?.pixels
  if (!pixels || pixels.length === 0) {
    return {
      rangeMin: 0,
      rangeMax: 1,
      rawMin: 0,
      rawMax: 1,
      stretchMin: 0,
      stretchMax: 1,
    }
  }

  let rawMin = Number.POSITIVE_INFINITY
  let rawMax = Number.NEGATIVE_INFINITY
  for (let index = 0; index < pixels.length; index += 1) {
    const value = Number(pixels[index])
    if (!Number.isFinite(value)) {
      continue
    }
    if (value < rawMin) rawMin = value
    if (value > rawMax) rawMax = value
  }

  if (!Number.isFinite(rawMin) || !Number.isFinite(rawMax)) {
    rawMin = 0
    rawMax = 1
  }
  if (rawMax <= rawMin) {
    rawMax = rawMin + 1
  }

  const safeMin = Number.isFinite(stretchMin) ? stretchMin : rawMin
  const safeMax = Number.isFinite(stretchMax) ? stretchMax : rawMax
  const nextStretchMin = Math.min(safeMin, safeMax)
  const nextStretchMax = Math.max(safeMin, safeMax)
  const rangeMin = Math.min(rawMin, nextStretchMin, nextStretchMax)
  const rangeMax = Math.max(rawMax, nextStretchMin, nextStretchMax)

  return {
    rawMin,
    rawMax,
    rangeMin,
    rangeMax: rangeMax > rangeMin ? rangeMax : rangeMin + 1,
    stretchMin: nextStretchMin,
    stretchMax: nextStretchMax > nextStretchMin ? nextStretchMax : nextStretchMin + 1,
  }
}

export function buildBrightnessMatchViewStatesByView(nodesByView, options = {}) {
  const states = {}
  for (const [view, node] of Object.entries(nodesByView || {})) {
    if (!node?.pixels || node.pixels.length === 0) {
      continue
    }
    const interval = computeBrightnessMatchIntervalFromPixels(node.pixels, options)
    states[view] = buildViewStretchState(node, interval.displayMin, interval.displayMax)
  }
  return states
}

export function buildFullRangeViewStatesByView(nodesByView) {
  const states = {}
  for (const [view, node] of Object.entries(nodesByView || {})) {
    if (!node?.pixels || node.pixels.length === 0) {
      continue
    }
    states[view] = buildViewStretchState(node, undefined, undefined)
  }
  return states
}

export function matchViewStatesFromSourceState(nodesByView, sourceView, sourceState, options = {}) {
  const sourceNode = nodesByView?.[sourceView]
  if (!sourceNode?.pixels || !isFiniteNumber(sourceState?.stretchMin) || !isFiniteNumber(sourceState?.stretchMax)) {
    throw new Error('source view stretch state is incomplete')
  }

  const inferred = inferMatchPositionsFromTargetInterval(
    sourceNode.pixels,
    Number(sourceState.stretchMin),
    Number(sourceState.stretchMax),
    options,
  )

  const states = {}
  for (const [view, node] of Object.entries(nodesByView || {})) {
    if (!node?.pixels || node.pixels.length === 0) {
      continue
    }
    if (view === sourceView) {
      states[view] = buildViewStretchState(node, sourceState.stretchMin, sourceState.stretchMax)
      continue
    }
    const interval = computeBrightnessMatchIntervalFromPixels(node.pixels, {
      ...options,
      backgroundPosition: inferred.backgroundPosition,
      highlightPosition: inferred.highlightPosition,
    })
    states[view] = buildViewStretchState(node, interval.displayMin, interval.displayMax)
  }
  return states
}

export const DEFAULT_BRIGHTNESS_MATCH_OPTIONS = {
  maxSamples: 200000,
  highPercentile: 99.9,
  highlightSigma: 5.0,
  backgroundPosition: 0.10,
  highlightPosition: 0.98,
  adaptiveHighPercentile: false,
}

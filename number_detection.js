let model
const checkButton = document.getElementById('save')
const canvas = document.querySelector('#canvas')
const smallCanvas = document.querySelector('#smallCanvas')
const ctx = canvas.getContext('2d')
const clearButton = document.getElementById('clear')
const smallCtx = smallCanvas.getContext('2d')
const result = document.getElementById('number')

const divide = num => parseFloat(num / 255)

function checkImage() {
	resample_single(canvas, 28, 28, smallCanvas)

	var imgData = smallCtx.getImageData(0, 0, 28, 28)
	var arr = []
	var arr28 = []
	for (var p = 0, i = 0; p < imgData.data.length; p += 4) {
		var valor = imgData.data[p + 3] / 255
		arr28.push([valor])
		if (arr28.length == 28) {
			arr.push(arr28)
			arr28 = []
		}
	}

	arr = [arr]
	var tensor4 = tf.tensor4d(arr)
	var resultados = model.predict(tensor4).dataSync()
	var mayorIndice = resultados.indexOf(Math.max.apply(null, resultados))

	console.log('Prediccion', mayorIndice)
	result.innerHTML = mayorIndice
}

async function loadModel() {
	model = await tf.loadLayersModel('/model/model.json')
	console.log('model loaded')
}

function clear() {
	ctx.clearRect(0, 0, canvas.width, canvas.height)
	result.innerHTML = ''
}

// Source: https://www.geeksforgeeks.org/how-to-draw-with-mouse-in-html-5-canvas/
window.addEventListener('load', () => {
	loadModel()
	document.addEventListener('mousedown', startPainting)
	document.addEventListener('mouseup', stopPainting)
	document.addEventListener('mousemove', sketch)
	checkButton.addEventListener('click', checkImage)
	clearButton.addEventListener('click', clear)

	canvas.addEventListener(
		'touchmove',
		function (e) {
			var touch = e.touches[0]
			var mouseEvent = new MouseEvent('mousemove', {
				clientX: touch.clientX,
				clientY: touch.clientY,
			})
			canvas.dispatchEvent(mouseEvent)
		},
		false
	)
})

let coord = { x: 0, y: 0 }
let paint = false

function getPosition(event) {
	event.preventDefault()
	event.stopPropagation()
	coord.x = event.clientX - canvas.offsetLeft
	coord.y = event.clientY - canvas.offsetTop
}

function startPainting(event) {
	event.preventDefault()
	event.stopPropagation()
	paint = true
	getPosition(event)
}
function stopPainting() {
	paint = false
}

function sketch(event) {
	event.preventDefault()
	event.stopPropagation()
	if (!paint) return
	ctx.beginPath()
	ctx.lineWidth = 10
	ctx.lineCap = 'round'
	ctx.strokeStyle = '#000'
	ctx.moveTo(coord.x, coord.y)
	getPosition(event)
	ctx.lineTo(coord.x, coord.y)
	ctx.stroke()
}

/**
 * Hermite resize - fast image resize/resample using Hermite filter. 1 cpu version!
 * Source: RingaTech github
 * @param {HtmlElement} canvas
 * @param {int} width
 * @param {int} height
 * @param {boolean} resize_canvas if true, canvas will be resized. Optional.
 * Cambiado por RT, resize canvas ahora es donde se pone el chiqitillllllo
 */
function resample_single(canvas, width, height, resize_canvas) {
	var width_source = canvas.width
	var height_source = canvas.height
	width = Math.round(width)
	height = Math.round(height)

	var ratio_w = width_source / width
	var ratio_h = height_source / height
	var ratio_w_half = Math.ceil(ratio_w / 2)
	var ratio_h_half = Math.ceil(ratio_h / 2)

	var ctx = canvas.getContext('2d')
	var ctx2 = resize_canvas.getContext('2d')
	var img = ctx.getImageData(0, 0, width_source, height_source)
	var img2 = ctx2.createImageData(width, height)
	var data = img.data
	var data2 = img2.data

	for (var j = 0; j < height; j++) {
		for (var i = 0; i < width; i++) {
			var x2 = (i + j * width) * 4
			var weight = 0
			var weights = 0
			var weights_alpha = 0
			var gx_r = 0
			var gx_g = 0
			var gx_b = 0
			var gx_a = 0
			var center_y = (j + 0.5) * ratio_h
			var yy_start = Math.floor(j * ratio_h)
			var yy_stop = Math.ceil((j + 1) * ratio_h)
			for (var yy = yy_start; yy < yy_stop; yy++) {
				var dy = Math.abs(center_y - (yy + 0.5)) / ratio_h_half
				var center_x = (i + 0.5) * ratio_w
				var w0 = dy * dy //pre-calc part of w
				var xx_start = Math.floor(i * ratio_w)
				var xx_stop = Math.ceil((i + 1) * ratio_w)
				for (var xx = xx_start; xx < xx_stop; xx++) {
					var dx = Math.abs(center_x - (xx + 0.5)) / ratio_w_half
					var w = Math.sqrt(w0 + dx * dx)
					if (w >= 1) {
						//pixel too far
						continue
					}
					//hermite filter
					weight = 2 * w * w * w - 3 * w * w + 1
					var pos_x = 4 * (xx + yy * width_source)
					//alpha
					gx_a += weight * data[pos_x + 3]
					weights_alpha += weight
					//colors
					if (data[pos_x + 3] < 255) weight = (weight * data[pos_x + 3]) / 250
					gx_r += weight * data[pos_x]
					gx_g += weight * data[pos_x + 1]
					gx_b += weight * data[pos_x + 2]
					weights += weight
				}
			}
			data2[x2] = gx_r / weights
			data2[x2 + 1] = gx_g / weights
			data2[x2 + 2] = gx_b / weights
			data2[x2 + 3] = gx_a / weights_alpha
		}
	}

	//Ya que esta, exagerarlo. Blancos blancos y negros negros..?

	for (var p = 0; p < data2.length; p += 4) {
		var gris = data2[p] //Esta en blanco y negro

		if (gris < 100) {
			gris = 0 //exagerarlo
		} else {
			gris = 255 //al infinito
		}

		data2[p] = gris
		data2[p + 1] = gris
		data2[p + 2] = gris
	}

	ctx2.putImageData(img2, 0, 0)
}

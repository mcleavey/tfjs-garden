import * as tf from '@tensorflow/tfjs';



sound.play();

const controls = ['up', 'down', 'left', 'right'];
const control_codes = [38, 40, 37, 39];


export function init() {
	document.getElementById('controller').style.display = '';
	statusElement.style.display = 'none';
}

const trainStatusElement = document.getElementById('train-status');
const statusElement = document.getElementById('status');

const learningRateElement = document.getElementById('learningRate');
export const getLearningRate = () => +learningRateElement.value;

const batchSizeFractionElement = document.getElementById('batchSizeFraction');
export const getBatchSizeFraction = () => +batchSizeFractionElement.value;

const epochsElement = document.getElementById('epochs');
export const getEpochs = () => +epochsElement.value;

const denseUnitsElement = document.getElementById('dense-units');
export const getDenseUnits = () => +denseUnitsElement.value;

export function startPacman() {

}

export function stopPacman() {

}

export function predictClass(classID) {

	document.body.setAttribute('data-active', controls[classID]);
	console.log("Activate " + classID);
}

export function isPredicting() {
	statusElement.style.visibility = 'visible';
}

export function donePredicting() {
	statusElement.style.visibility = 'hidden';
}

export function trainStatus(message) {
	trainStatusElement.innerText = message;
}

export let addExampleHandler;
export function setExampleHandler(handler) {
	addExampleHandler = handler;
}


let mouseDown = false;
let totals = [0,0,0,0];
const upButton = document.getElementById('up');
const downButton = document.getElementById('down');
const leftButton = document.getElementById('left');
const rightButton = document.getElementById('right');

const thumbDisplayed = {};


async function handler(classID) {
	mouseDown = true;
	const className = controls[classID];
	const button = document.getElementById(className);
	const total = document.getElementById(className+'-total');
	while (mouseDown) {
		addExampleHandler(classID);
		document.body.setAttribute('data-active', controls[classID]);
		total.innerText = totals[classID]++; 
		await tf.nextFrame();		
	}

	document.body.removeAttribute('data-active');
}


upButton.addEventListener('mousedown', () => handler(0));
upButton.addEventListener('mouseup', () => mouseDown=false);

downButton.addEventListener('mousedown', () => handler(1));
downButton.addEventListener('mouseup', () => mouseDown=false);

leftButton.addEventListener('mousedown', () => handler(2));
leftButton.addEventListener('mouseup', () => mouseDown=false);

rightButton.addEventListener('mousedown', () => handler(3));
rightButton.addEventListener('mouseup', () => mouseDown=false);


export function drawThumb(img, label) {
	if (thumbDisplayed[label]==null) {
		const thumbCanvas = document.getElementById(controls[label]+'-thumb');
		draw(img, thumbCanvas);
	}
}

export function draw(img, canvas) {
	const [width, height] = [224, 224];
	const ctx = canvas.getContext('2d');
	const imageData = new ImageData(width, height);
    const data = img.dataSync();

    for (let i=0; i<width*height; ++i) {
    	const j = i*4;
    	imageData.data[j+0] = (data[i*3+0] + 1 ) * 127;
    	imageData.data[j+1] = (data[i*3+1] + 1 ) * 127;
    	imageData.data[j+2] = (data[i*3+2] + 1 ) * 127;
    	imageData.data[j+3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
}

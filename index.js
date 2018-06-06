import * as tf from '@tensorflow/tfjs';
import {Webcam} from './webcam';
import * as ui from './ui';
import {ControllerDataset} from './controller_dataset.js';
import {Howl, Howler} from 'howler';

const sound = new Howl({
	src: ['./sounds/bubbles.mp3']
});

var audio = new Audio('images/bubbles.mp3');
audio.play();


const NUM_CLASSES = 4;
const webcam = new Webcam(document.getElementById('webcam'));
const controllerDataset = new ControllerDataset(NUM_CLASSES);
let mobilenet;

async function loadMobilenet(){
	const mobilenet = await tf.loadModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
	const layer = mobilenet.getLayer('conv_pw_13_relu');
	return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

ui.setExampleHandler(label => {
    tf.tidy(() => {
    	const img = webcam.capture();
    	// add image to training
    	controllerDataset.addExample(mobilenet.predict(img), label);
    	ui.drawThumb(img, label);
    });
});


let isPredicting = false;

document.getElementById('train').addEventListener('click', async () => {
	ui.trainStatus('Training...');
	await tf.nextFrame();
	await tf.nextFrame();	
	isPredicting = false;
	train();
});

document.getElementById('predict').addEventListener('click', async() => {
	ui.startPacman();
	isPredicting = true;
	predict();
});


document.getElementById('stop').addEventListener('click', async() => {
	ui.stopPacman();
	isPredicting = false;
});

async function init() {
	await webcam.setup();
	mobilenet = await loadMobilenet();
	ui.init();
}

init();



///  TRAINING AND PREDICTING FROM MODEL

let model;
async function train() {
	if (controllerDataset.xs == null) {
		throw new Error('No training examples yet!');
	}

	model = tf.sequential({
		layers: [
			tf.layers.flatten({inputShape: [7, 7, 256]}),
			tf.layers.dense({
				units: ui.getDenseUnits(),
				activation: 'relu',
				kernelInitializer: 'varianceScaling',
				useBias: true
			}),
			tf.layers.dense({
				units: NUM_CLASSES,
				activation: 'softmax',
				kernelInitializer: 'varianceScaling',
				useBias: false
			})
		]
	});

	const optimizer = tf.train.adam(ui.getLearningRate());
	model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});

	const bs = Math.floor(controllerDataset.xs.shape[0] * ui.getBatchSizeFraction());
	if (!(bs>0)) {
		throw new Error ('Batch size is 0 or NaN');
	}

	model.fit(controllerDataset.xs, controllerDataset.ys, {
		bs,
		epochs: ui.getEpochs(),
		callbacks: {
			onBatchEnd: async(batch, logs) => {
				ui.trainStatus('Loss: ' + logs.loss.toFixed(5));
				await tf.nextFrame();
			}
		}
	});
}


async function predict() {
	ui.isPredicting();
	while (isPredicting) {
		const predictedClass = tf.tidy(() => {
			const img = webcam.capture();
			const mobilenet_activations = mobilenet.predict(img);
			const movement_pred = model.predict(mobilenet_activations);
			return movement_pred.as1D().argMax();
		});
		const classId = (await predictedClass.data())[0];
		predictedClass.dispose();

		ui.predictClass(classId);
		await tf.nextFrame();
	}
	ui.donePredicting();
}
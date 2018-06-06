import * as tf from '@tensorflow/tfjs';

export class ControllerDataset {
	constructor(numClasses) {
		this.numClasses = numClasses;
	}

	addExample(example, label) {
		const y = tf.tidy(() => 
			tf.oneHot(tf.tensor1d([label]).toInt(), this.numClasses)
		);

		if (this.xs == null) {
			// first training example
			this.xs = tf.keep(example);
			this.ys = tf.keep(y);
		} else {
			// add to already existing examples
			const old_xs = this.xs;
			const old_ys = this.ys;

			this.xs = tf.keep(old_xs.concat(example, 0));
			this.ys = tf.keep(old_ys.concat(y, 0));

			old_xs.dispose();
			old_ys.dispose();
			y.dispose();
		}
	}
}
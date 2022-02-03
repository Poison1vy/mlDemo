// create the model
const model = tf.sequential();

// create input and hidden layer
const hidden = tf.layers.dense({
    units: 4, 
    inputShape: [1], 
    activation: 'sigmoid'
});

// create output layer
const output = tf.layers.dense({
    units: 1, 
    activation: 'sigmoid'
});

// add both layers to the model
model.add(hidden); 
model.add(output); 

// compile the model
model.compile({
    optimizer: tf.train.sgd(1), 
    loss: tf.losses.meanSquaredError
});

// inputs
let inputs = tf.tensor2d([
    [0],
    [0.5],
    [1]
]);

// outputs
let answers = tf.tensor2d([
    [1],
    [0.5],
    [0]
]);

// config
const config = {
    epochs: 5,
    shuffle: true
};

// training function
async function train() {
    for (let i = 0; i < 100; i++) {
        const response = await model.fit(inputs, answers, config);
        console.log(response.history.loss[0]);
    }
}

// predict at the start
model.predict(inputs).print();

// train, then predict again
train().then(() => {
    console.log('training is complete');
    model.predict(tf.tensor2d([[0.2], [0.8], [1], [0.5], [0.1], [20]])).print();
});
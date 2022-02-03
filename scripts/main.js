let xVals = [];
let yVals = [];
let points = [];

let m, b;

const learningRate = 0.5;


function setup() {
    createCanvas(windowWidth - 20, windowHeight - 20);

    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function draw() {
    background(15);

    for (pt of points) {
        noStroke();
        fill(200);
        circle(pt[0], pt[1], 10);
    }

    if (xVals.length > 0) {
        optimizer.minimize(() => {
            return loss(predict(xVals), tf.tensor1d(yVals));
        });
    }

    const prediction = predict([0, 1]);
    result = prediction.dataSync()

    stroke(255);
    line(0, map(result[0], 0, 1, 0, height), width, map(result[1], 0, 1, 0, height));
}

function mousePressed() {
    xVals.push(map(mouseX, 0, width, 0, 1));
    yVals.push(map(mouseY, 0, height, 0, 1));
    points.push([mouseX, mouseY]);
}

const optimizer = tf.train.sgd(learningRate);

function predict(x) {
    const xs = tf.tensor1d(x);

    return xs.mul(m).add(b);
}

function loss(predictions, labels) {
    return predictions.sub(labels).square().mean()
}
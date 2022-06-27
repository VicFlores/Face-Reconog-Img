const imageUpload = document.getElementById('imageUpload');
const imgContainer = document.getElementById('imgContainer');
const img = document.getElementById('img');
const width = '300';
const height = '350';
let faceMatcher;

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
]);

async function loadImage() {
  const image = await faceapi.bufferToImage(imageUpload.files[0]);
  const canvas = faceapi.createCanvasFromMedia(image);
  const displaySize = { width: 300, height: 350 };

  img.src = image.src;
  imgContainer.append(canvas);

  faceapi.matchDimensions(canvas, displaySize);

  const detections = await faceapi
    .detectAllFaces(image)
    .withFaceLandmarks()
    .withFaceDescriptors();

  const resizeDetection = faceapi.resizeResults(detections, displaySize);

  resizeDetection.map((strike) => {
    const box = strike.detection.box;
    const drawBox = new faceapi.draw.DrawBox(box, { label: 'Face' });
    drawBox.draw(canvas);
  });
}

function loadLabelImages() {
  const labels = [
    'Black Widow',
    'Captain America',
    'Captain Marvel',
    'Hawkeye',
    'Jim Rhodes',
    'Thor',
    'Tony Stark',
  ];

  return Promise.all(
    labels.map(async (label) => {
      const descriptions = [];

      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `https://github.com/WebDevSimplified/Face-Recognition-JavaScript/tree/master/labeled_images/${label}/${i}.jpg `
        );

        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptors();

        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
  );
}

async function run() {
  const labeledFaceDescriptors = await loadLabelImages();
  faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);
}

run();

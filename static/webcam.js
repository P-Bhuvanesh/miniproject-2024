// document.addEventListener('DOMContentLoaded', function () {
//     const videoElement = document.getElementById('webcam-feed');
//     const startButton = document.getElementById('start-btn');
//     const stopButton = document.getElementById('stop-btn');
//     let intervalID = null;

//     // Function to start video feed
//     function startVideoFeed() {
//         // videoElement.src = '/video_feed';
//         // startButton.disabled = true;
//         // stopButton.disabled = false;

//         let triggerVideo = document.getElementById("webcam-feed");
//         triggerVideo.src = "{{ url_for('video_feed') }}";
//         // Continuously update the feed every second (1 second delay)
//         intervalID = setInterval(function() {
//             videoElement.src = '/video_feed?' + new Date().getTime(); // Add timestamp to avoid caching
//         }, 1000);
//     }

//     // Function to stop video feed
//     function stopVideoFeed() {
//         clearInterval(intervalID); // Stop the interval
//         videoElement.src = ''; // Remove the feed
//         startButton.disabled = false;
//         stopButton.disabled = true;
//     }

//     // Add event listeners to buttons
//     startButton.addEventListener('click', startVideoFeed);
//     stopButton.addEventListener('click', stopVideoFeed);
// });

let intervalId = null;

let startVideoBtn = document.getElementById("start-btn");
startVideoBtn.addEventListener("click", () => {
    startVideoBtn.style.backgroundColor="red";
});


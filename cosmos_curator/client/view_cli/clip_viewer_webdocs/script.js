async function fetchClipNames() {
    const response = await fetch('/list_clips');
    console.log('Fetching clip names via HTTP from:', 'http://localhost:8080/list_clips');
    if (!response.ok) {
        throw new Error('Failed to fetch clip names');
    }
    return response.json();
}

function formatTimestamp(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
}

async function createVideoElement(videoPath) {
    const video = document.createElement('video');
    video.src = videoPath;
    video.controls = true;
    video.autoplay = true;
    video.loop = true;
    video.muted = true;
    video.playsInline = true;

    // Add event listener to handle video loading
    video.addEventListener('loadedmetadata', () => {
        const parent = video.parentElement;
        if (parent) {
            // Adjust container size based on video aspect ratio
            const videoAspect = video.videoWidth / video.videoHeight;
            const containerAspect = parent.clientWidth / parent.clientHeight;

            if (videoAspect > containerAspect) {
                video.style.width = '100%';
                video.style.height = 'auto';
            } else {
                video.style.width = 'auto';
                video.style.height = '100%';
            }
        }
    });

    return video;
}

function createTimestampElement(startTime, endTime) {
    const container = document.createElement('div');
    container.className = 'timestamp-container';

    // Create timestamp content
    const content = document.createElement('div');
    content.className = 'timestamp-content';

    // Add label
    const label = document.createElement('span');
    label.className = 'timestamp-label';
    label.textContent = 'Time Range:';
    content.appendChild(label);

    // Add start time
    const startTimeSpan = document.createElement('span');
    startTimeSpan.className = 'timestamp-value';
    startTimeSpan.textContent = formatTimestamp(startTime);
    content.appendChild(startTimeSpan);

    // Add separator
    const separator = document.createElement('span');
    separator.className = 'timestamp-separator';
    separator.textContent = '→';
    content.appendChild(separator);

    // Add end time
    const endTimeSpan = document.createElement('span');
    endTimeSpan.className = 'timestamp-value';
    endTimeSpan.textContent = formatTimestamp(endTime);
    content.appendChild(endTimeSpan);

    // Add clock emoji
    const icon = document.createElement('span');
    icon.className = 'timestamp-icon';
    icon.textContent = '🕒';
    container.appendChild(icon);

    // Add the content
    container.appendChild(content);

    return container;
}

(async function() {
    try {
        const content = document.getElementById('content');
        const clipNames = await fetchClipNames();

        for (const clipName of clipNames) {
            const videoPath = `clips/${clipName}.mp4`;
            const jsonPath = `metas/v0/${clipName}.json`;

            console.log('Fetching video via HTTP from:', `http://localhost:8080/${videoPath}`);
            const video = await createVideoElement(videoPath);

            // Create clip container
            const clipContainer = document.createElement('div');
            clipContainer.className = 'clip-row';

            // Create and setup video container
            const videoContainer = document.createElement('div');
            videoContainer.className = 'video-container';
            videoContainer.appendChild(video);

            // Create and setup caption container
            const captionContainer = document.createElement('div');
            captionContainer.className = 'caption-container';

            // Create caption content wrapper
            const captionContent = document.createElement('div');
            captionContent.className = 'caption-content';

            // Fetch metadata for caption and timestamps
            let captionText = "No caption available";
            try {
                console.log('Fetching metadata via HTTP from:', `http://localhost:8080/${jsonPath}`);
                const response = await fetch(jsonPath);
                if (response.ok) {
                    const jsonData = await response.json();
                    // Get caption
                    if (jsonData.windows?.[0]?.cosmos_r2_caption) {
                        captionText = jsonData.windows[0].cosmos_r2_caption;
                    } else if (jsonData.windows?.[0]?.cosmos_r1_caption) {
                        captionText = jsonData.windows[0].cosmos_r1_caption;
                    } else if (jsonData.windows?.[0]?.gemini_caption) {
                        captionText = jsonData.windows[0].gemini_caption;
                    } else if (jsonData.windows?.[0]?.nemotron_caption) {
                        captionText = jsonData.windows[0].nemotron_caption;
                    } else if (jsonData.windows?.[0]?.qwen_caption) {
                        captionText = jsonData.windows[0].qwen_caption;
                    }
                    // Get and add timestamp
                    if (jsonData.duration_span) {
                        const [startTime, endTime] = jsonData.duration_span;
                        const timestamp = createTimestampElement(startTime, endTime);
                        videoContainer.appendChild(timestamp);
                    }
                }
            } catch (e) {
                console.warn(`Could not load metadata for ${clipName}:`, e);
            }

            const title = document.createElement('h2');
            title.textContent = "Caption";

            const paragraph = document.createElement('div');
            paragraph.innerHTML = marked.parse(captionText);

            captionContent.appendChild(title);
            captionContent.appendChild(paragraph);
            captionContainer.appendChild(captionContent);

            // Assemble the clip container
            clipContainer.appendChild(videoContainer);
            clipContainer.appendChild(captionContainer);
            content.appendChild(clipContainer);
        }
    } catch (error) {
        console.error('Error loading clips:', error);
        const errorMessage = document.createElement('div');
        errorMessage.textContent = 'Error loading clips. Please try again later.';
        errorMessage.style.color = 'red';
        document.getElementById('content').appendChild(errorMessage);
    }
})();

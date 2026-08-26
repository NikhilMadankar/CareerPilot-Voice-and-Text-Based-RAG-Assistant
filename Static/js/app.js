let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

$(document).ready(function() {

    // Configure marked options
    if (window.marked) {
        marked.setOptions({
            breaks: true,
            gfm: true
        });
    }

    function stopCurrentSpeech() {
        const ttsAudio = document.getElementById("ttsAudio");
        if (ttsAudio) {
            ttsAudio.pause();
            ttsAudio.currentTime = 0;
        }
        $("#stopSpeaking").hide();
    }

    function renderMessage(role, text, sources = []) {
        const isUser = role === "user";
        const avatarIcon = isUser ? "fa-user" : "fa-robot";
        const parsedContent = isUser ? escapeHtml(text) : (window.marked ? marked.parse(text) : text);

        let sourcesHtml = "";
        if (!isUser && sources && sources.length > 0) {
            const unique = [];
            const seen = new Set();
            sources.forEach(s => {
                const key = `${s.course_name}|${s.page_number}`;
                if (!seen.has(key)) {
                    seen.add(key);
                    unique.push(s);
                }
            });

            sourcesHtml = '<div class="sources-container">';
            unique.forEach(src => {
                const course = src.course_name || "Course Guide";
                const cat = src.category || "General";
                const page = src.page_number ? `p. ${src.page_number}` : "";
                sourcesHtml += `
                    <div class="source-tag">
                        <i class="fa-solid fa-book-open"></i>
                        <span><strong>${escapeHtml(course)}</strong> (${escapeHtml(cat)}${page ? ' • ' + page : ''})</span>
                    </div>
                `;
            });
            sourcesHtml += '</div>';
        }

        const msgHtml = `
            <div class="message-row ${isUser ? 'user' : 'ai'}">
                <div class="avatar"><i class="fa-solid ${avatarIcon}"></i></div>
                <div class="bubble">
                    ${parsedContent}
                    ${sourcesHtml}
                </div>
            </div>
        `;

        $("#chatOutput").append(msgHtml);
        scrollToBottom();
    }

    function scrollToBottom() {
        const chatBox = document.getElementById("chatOutput");
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function escapeHtml(text) {
        if (!text) return "";
        return $('<div>').text(text).html();
    }

    // ---------------- Text Chat ----------------
    async function handleSendText() {
        const query = $("#textQuery").val().trim();
        if (!query) return;

        // Stop any currently-playing bot speech
        stopCurrentSpeech();

        renderMessage("user", query);
        $("#textQuery").val("");
        $("#sendTextBtn").prop("disabled", true);

        // Show typing placeholder
        const typingId = "typing-" + Date.now();
        $("#chatOutput").append(`
            <div class="message-row ai" id="${typingId}">
                <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                <div class="bubble"><i class="fa-solid fa-spinner fa-spin"></i> Consulting NCERT career guidelines...</div>
            </div>
        `);
        scrollToBottom();

        try {
            const res = await fetch("/chat", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({query})
            });

            const data = await res.json();
            $(`#${typingId}`).remove();

            if (data.error) {
                renderMessage("ai", `⚠️ **Error:** ${data.error}`);
            } else {
                renderMessage("ai", data.answer, data.sources);
            }
        } catch (err) {
            $(`#${typingId}`).remove();
            renderMessage("ai", `⚠️ **Network error:** Could not reach the CareerGuide AI server.`);
        } finally {
            $("#sendTextBtn").prop("disabled", false);
            $("#textQuery").focus();
        }
    }

    $("#sendTextBtn").click(handleSendText);

    $("#textQuery").keypress(function(e) {
        if (e.which === 13) {
            e.preventDefault();
            handleSendText();
        }
    });

    // ---------------- Suggestion Chips ----------------
    $(".suggestion-chip").click(function() {
        const q = $(this).data("query");
        if (q) {
            $("#textQuery").val(q);
            handleSendText();
        }
    });

    // ---------------- Stop Speaking Button ----------------
    $("#stopSpeaking").click(function() {
        stopCurrentSpeech();
    });

    // Auto-hide Stop Speaking button when audio ends
    const ttsAudio = document.getElementById("ttsAudio");
    if (ttsAudio) {
        ttsAudio.addEventListener("ended", function() {
            $("#stopSpeaking").hide();
        });
        ttsAudio.addEventListener("pause", function() {
            $("#stopSpeaking").hide();
        });
    }

    // ---------------- Voice Chat (Recording) ----------------
    $("#voiceBtn").click(async function() {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    });

    async function startRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert("Microphone access is not supported in this browser.");
            return;
        }

        // Stop any currently-playing bot speech before starting a new recording
        stopCurrentSpeech();

        try {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async function() {
                // Stop audio tracks
                stream.getTracks().forEach(track => track.stop());

                $("#voiceStatus").removeClass("active");
                $("#statusText").text("Processing speech...");
                $("#voiceStatus").addClass("active");

                const blob = new Blob(audioChunks, { type: "audio/wav" });
                const formData = new FormData();
                formData.append("file", blob, "user_voice.wav");

                const typingId = "voice-typing-" + Date.now();
                $("#chatOutput").append(`
                    <div class="message-row ai" id="${typingId}">
                        <div class="avatar"><i class="fa-solid fa-robot"></i></div>
                        <div class="bubble"><i class="fa-solid fa-spinner fa-spin"></i> Transcribing & processing query...</div>
                    </div>
                `);
                scrollToBottom();

                try {
                    const response = await fetch("/voice", { method: "POST", body: formData });
                    const data = await response.json();
                    $(`#${typingId}`).remove();
                    $("#voiceStatus").removeClass("active");

                    if (data.error) {
                        renderMessage("ai", `⚠️ **Voice Error:** ${data.error}`);
                    } else {
                        renderMessage("user", `🎙️ ${data.query}`);
                        renderMessage("ai", data.answer, data.sources);

                        // Play sanitized TTS audio
                        if (data.tts_audio_path) {
                            const audioElem = document.getElementById("ttsAudio");
                            audioElem.src = data.tts_audio_path + "?cache=" + new Date().getTime();
                            $("#stopSpeaking").css("display", "inline-flex");
                            audioElem.play().catch(e => console.log("Audio autoplay prevented by browser policy", e));
                        }
                    }
                } catch (err) {
                    $(`#${typingId}`).remove();
                    $("#voiceStatus").removeClass("active");
                    renderMessage("ai", "⚠️ Failed to process audio recording.");
                }
            };

            mediaRecorder.start();
            isRecording = true;
            $("#voiceBtn").addClass("recording");
            $("#micIcon").removeClass("fa-microphone").addClass("fa-stop");
            $("#micText").text("Stop");
            $("#statusText").text("Listening... Click Stop when finished speaking.");
            $("#voiceStatus").addClass("active");

        } catch (err) {
            console.error("Microphone access error:", err);
            alert("Could not access microphone. Please ensure microphone permissions are granted.");
        }
    }

    function stopRecording() {
        if (mediaRecorder && isRecording) {
            mediaRecorder.stop();
            isRecording = false;
            $("#voiceBtn").removeClass("recording");
            $("#micIcon").removeClass("fa-stop").addClass("fa-microphone");
            $("#micText").text("Voice");
        }
    }

});

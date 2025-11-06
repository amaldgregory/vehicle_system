// static/upload.js
const imageInput = document.getElementById("imageInput");
const preview = document.getElementById("preview");
const uploadBtn = document.getElementById("uploadBtn");
const resultDiv = document.getElementById("result");

imageInput.addEventListener("change", (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  preview.src = url;
});

uploadBtn.addEventListener("click", async () => {
  const file = imageInput.files[0];
  if (!file) {
    resultDiv.textContent = "Pick an image first.";
    return;
  }

  resultDiv.textContent = "Uploading...";
  const fd = new FormData();
  fd.append("image", file);

  try {
    const resp = await fetch("/api/upload", {
      method: "POST",
      body: fd
    });
    const data = await resp.json();
    if (!resp.ok) {
      resultDiv.textContent = "Error: " + (data.error || JSON.stringify(data));
      return;
    }

    let html = `<strong>OCR result:</strong> ${data.plate_raw || "(none)"}<br/>`;
    html += `<strong>Normalized:</strong> ${data.plate_normalized || "(none)"}<br/>`;
    html += `<strong>Banned:</strong> ${data.banned ? "YES" : "NO"}<br/>`;
    if (data.banned) {
      html += `<strong>Email sent:</strong> ${data.email_sent ? "YES" : "NO"}<br/>`;
      if (data.email_error) html += `<small>Error: ${data.email_error}</small><br/>`;
    }
    resultDiv.innerHTML = html;
  } catch (err) {
    resultDiv.textContent = "Network error: " + err.message;
  }
});

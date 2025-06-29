const express = require('express');
const cors = require('cors');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

const app = express();
const port = process.env.PORT || 3001;

// Supabase configuration
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_ANON_KEY;
const supabase = createClient(supabaseUrl, supabaseKey);

// Configure multer for file uploads
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    const uploadDir = path.join(__dirname, '../uploads');
    if (!fs.existsSync(uploadDir)) {
      fs.mkdirSync(uploadDir, { recursive: true });
    }
    cb(null, uploadDir);
  },
  filename: function (req, file, cb) {
    const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
    cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
  }
});

const upload = multer({ 
  storage: storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: function (req, file, cb) {
    if (file.fieldname === 'audio') {
      if (file.mimetype.startsWith('audio/')) {
        cb(null, true);
      } else {
        cb(new Error('Only audio files are allowed'));
      }
    } else if (file.fieldname === 'image') {
      if (file.mimetype.startsWith('image/')) {
        cb(null, true);
      } else {
        cb(new Error('Only image files are allowed'));
      }
    } else {
      cb(new Error('Unexpected field'));
    }
  }
});

// Middleware
app.use(cors());
app.use(express.json());

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ status: 'OK', message: 'Server is running' });
});

// API routes
app.get('/api/test', (req, res) => {
  res.json({ message: 'API is working' });
});

// Video generation endpoint
app.post('/api/generate-video', upload.fields([
  { name: 'audio', maxCount: 1 },
  { name: 'image', maxCount: 1 }
]), async (req, res) => {
  try {
    console.log('Video generation request received');
    console.log('Files:', req.files);
    console.log('Body:', req.body);

    // Validate required fields
    if (!req.files || !req.files.audio || !req.files.image) {
      return res.status(400).json({ error: 'Missing required files (audio and image)' });
    }

    if (!req.body.prompt || !req.body.userId) {
      return res.status(400).json({ error: 'Missing required fields (prompt and userId)' });
    }

    const { prompt, resolution, frameNum, userId } = req.body;
    const audioFile = req.files.audio[0];
    const imageFile = req.files.image[0];

    // Create unique job ID
    const jobId = Date.now() + '-' + Math.round(Math.random() * 1E9);
    
    // Create input JSON for the Python script
    const inputData = {
      prompt: prompt,
      cond_image: path.resolve(imageFile.path),
      cond_audio: {
        person1: path.resolve(audioFile.path)
      },
      audio_type: "add"
    };

    const inputJsonPath = path.join(__dirname, '../uploads', `input_${jobId}.json`);
    fs.writeFileSync(inputJsonPath, JSON.stringify(inputData, null, 2));

    // Create database record for the video generation job
    const { data: videoRecord, error: dbError } = await supabase
      .from('generated_videos')
      .insert([{
        user_id: userId,
        title: prompt.substring(0, 50),
        prompt: prompt,
        status: 'processing',
        resolution: resolution,
        frame_count: parseInt(frameNum),
        created_at: new Date().toISOString()
      }])
      .select()
      .single();

    if (dbError) {
      console.error('Database error:', dbError);
      return res.status(500).json({ error: 'Failed to create video record' });
    }

    console.log('Video record created:', videoRecord.id);

    // Return immediate response
    res.json({ 
      success: true, 
      jobId: jobId,
      videoId: videoRecord.id,
      message: 'Video generation started successfully' 
    });

  } catch (error) {
    console.error('Video generation error:', error);
    res.status(500).json({ error: 'Internal server error: ' + error.message });
  }
});

// Serve static files (generated videos)
app.use('/uploads', express.static(path.join(__dirname, '../uploads')));

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
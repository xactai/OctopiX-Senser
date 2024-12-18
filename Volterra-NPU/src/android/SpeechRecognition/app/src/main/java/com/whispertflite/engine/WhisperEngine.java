package com.whispertflite.engine;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public interface WhisperEngine {
    boolean isInitialized();
    boolean initialize(String modelPath, String vocabPath, boolean multilingual) throws IOException;
    void deinitialize();
    String transcribeFile(String wavePath);
    String transcribeBuffer(float[] samples);
}

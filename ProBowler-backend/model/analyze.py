from model.EnhancedCricketBiomechanicsExtractor import EnhancedCricketBiomechanicsExtractor

def analyze_bowling(video_path):
    """
    This function is called from app.py after the video is uploaded.
    It runs your biomechanics analysis.
    """
    print(f"Analyzing video: {video_path}")

    try:
        # Step 1: Initialize your feature extractor
        extractor = EnhancedCricketBiomechanicsExtractor(video_path)

        # Step 2: Call the correct analysis method
        result = extractor.calculate_biomechanical_summary()  # ✅ correct method name

        return {
            "status": "success",
            "message": "Video analyzed successfully",
            "details": result
        }

    except Exception as e:
        print("Error during analysis:", e)
        return {
            "status": "error",
            "message": str(e)
        }

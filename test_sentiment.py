import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.sentiment import SentimentAnalyzer

# 测试情感分析器
def test_sentiment_analyzer():
    print("Testing SentimentAnalyzer...")
    analyzer = SentimentAnalyzer()
    
    # 测试积极文本
    positive_text = "I love this movie! It's amazing!"
    positive_score = analyzer.score(positive_text)
    print(f"Positive text: '{positive_text}'")
    print(f"Score: {positive_score}")
    assert -1 <= positive_score <= 1, f"Score should be in [-1, 1], got {positive_score}"
    assert positive_score > 0, f"Positive text should have positive score, got {positive_score}"
    
    # 测试消极文本
    negative_text = "I hate this movie. It's terrible."
    negative_score = analyzer.score(negative_text)
    print(f"\nNegative text: '{negative_text}'")
    print(f"Score: {negative_score}")
    assert -1 <= negative_score <= 1, f"Score should be in [-1, 1], got {negative_score}"
    assert negative_score < 0, f"Negative text should have negative score, got {negative_score}"
    
    # 测试中性文本
    neutral_text = "This movie is okay."
    neutral_score = analyzer.score(neutral_text)
    print(f"\nNeutral text: '{neutral_text}'")
    print(f"Score: {neutral_score}")
    assert -1 <= neutral_score <= 1, f"Score should be in [-1, 1], got {neutral_score}"
    
    print("\n✅ All sentiment analyzer tests passed!")

if __name__ == "__main__":
    test_sentiment_analyzer()

from langdetect import detect

def detect_language(text):

    try:
        return detect(text)
    except:
        return "en"


def auto_reply(language,sentiment):

    if sentiment=="Negative":

        if language=="ta":
            return "உங்களுக்கு ஏற்பட்ட சிரமத்திற்கு மன்னிக்கவும்"

        elif language=="hi":
            return "असुविधा के लिए खेद है"

        else:
            return "We are sorry for the inconvenience"

    else:

        if language=="ta":
            return "உங்கள் கருத்துக்கு நன்றி"

        elif language=="hi":
            return "आपकी प्रतिक्रिया के लिए धन्यवाद"

        else:
            return "Thank you for your feedback"
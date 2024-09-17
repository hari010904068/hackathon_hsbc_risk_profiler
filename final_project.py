!pip install google-cloud-aiplatform
!gcloud auth application-default login
!pip install reportlab


project_id = "<your project id>"
input_prompt = """I am 35 year old with amount balance 250056 and with liabilities of home loan"""




import vertexai
from vertexai.language_models import ChatModel, InputOutputTextPair
from vertexai import generative_models
from vertexai.preview.vision_models import ImageGenerationModel
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch




vertexai.init(project=project_id, location="us-central1")
chat_model = ChatModel.from_pretrained("chat-bison")
parameters = {
    "candidate_count": 1,
    "max_output_tokens": 1024,
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1
}

safety_settings={
          generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
          generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    }
    
    
chat = chat_model.start_chat(
    context="""You are the risk profiler who categorizes data into [low,medium,high]
output should be in one word.""",
    examples=[
        InputOutputTextPair(
            input_text="""I am 46 year old with amount balance 25000 and no liabilities""",
            output_text="""low"""
        ),
        InputOutputTextPair(
            input_text="""I am 32 year old with amount balance 2500 and with liabilities of home loan""",
            output_text="""high"""
        ),
        InputOutputTextPair(
            input_text="""I am 45 year old with amount balance 32500 and with liabilities of home loan""",
            output_text="""medium"""
        ),
        InputOutputTextPair(
            input_text="""I am 18 year old with amount balance 2500 and with no liabilities""",
            output_text="""low"""
        )
    ]
)
response = chat.send_message(input_prompt, **parameters)
print(f"Response from Model: {response.text}")









profile = response.text




########################################


output_file = "my-output.png"
#prompt = "Equity stock" # The text prompt describing what you want to see.

def risk_profile(profile):
    switch={
      'high':'Equity',
      'medium':'Fixed Deposit',
      'low':'Savings Account',

      }
    return switch.get(profile,"Invalid input")

prompt = risk_profile(profile)

vertexai.init(project=project_id, location="us-central1")

model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-001")

images = model.generate_images(
    prompt=prompt,
    # Optional parameters
    number_of_images=1,
    language="en",
    # You can't use a seed value and watermark at the same time.
    # add_watermark=False,
    # seed=100,
    aspect_ratio="1:1",
    safety_filter_level="block_some",
    person_generation="allow_adult",
)

images[0].save(location=output_file, include_generation_parameters=False)

# Optional. View the generated image in a notebook.
# images[0].show()

print(f"Created output image using {len(images[0]._image_bytes)} bytes")


def pdf_gen(profile):

    # Create a PDF file
    pdf_filename = "Risk_Profile.pdf"
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    print(f"Created pdf with text")

    c.drawString(100, 750, "Hello, this is your risk profile . Your risk profile is "+profile + "You should invest in the following product  ")
    
    print(f"Created pdf with image")

    # Draw an image
    image_path = "my-output.png"  # Replace with your image path
    c.drawImage(image_path, 100, 600, width=2*inch, height=2*inch)

    # Save the PDF
    c.save()

pdf_gen(profile)
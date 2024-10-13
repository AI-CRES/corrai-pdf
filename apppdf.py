import streamlit as st
import openai
import base64
import pandas as pd
from io import BytesIO
from PIL import Image
import fitz  # PyMuPDF
import io
import re

hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode('utf-8')

def display_image_from_base64(encoded_image):
    image_data = base64.b64decode(encoded_image)
    image = Image.open(io.BytesIO(image_data))
    return image

# Si l'image contient des éléments non textuels, comme des cases à cocher, des réponses encerclées, ou d'autres éléments graphiques, décrivez-les de manière détaillée pour chaque question et indiqué qu'elle est cochée en utilisant le mot "coché" au debut.
def extract_content_from_image_reference(encoded_image, api_key,  vision_prompt):
    try:
        openai.api_key = api_key
        prompt = f"""
        Vous êtes un assistant qui identifie les noms des étudiants et identifie toutes  les questions , 
        reponse  et la ponderation (si ca existe) associer, en les reproduisant fidelement(sans analyser) comme transcrit sans analyser.
        Extraire  ou recuperer exactement ce qui se trouve sur l'image sans ajouter ni retrancher. en retranscrivant les contenues.
        les reponses peuvent manuscrit.
        
        Dire obligatoirement s'il y a des questions à choix multiples ou des questions de correspondance ou autres types des questions.
        
        Si le texte dans l'image ne peut pas être extrait directement, décrivez les éléments visuels présents, mais pas en detaille.
        
         {vision_prompt}
        
        """
        
        response = openai.ChatCompletion.create(
            model="chatgpt-4o-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            
        )
        content = response['choices'][0]['message']['content']
        return content
    except openai.error.OpenAIError as e:
        st.error(f"Erreur lors de la communication avec l'API OpenAI: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue: {e}")
        return None
    



# Si l'image contient des éléments non textuels, comme des cases à cocher, des réponses encerclées, ou d'autres éléments graphiques, décrivez-les de manière détaillée pour chaque question et indiqué qu'elle est cochée en utilisant le mot "coché" au debut.
def extract_content_from_image(encoded_image, api_key, vision_prompt, reference_content):
    try:
        openai.api_key = api_key
        prompt = f"""
        Vous êtes un assistant temporaire, Vous n'avez pas l'autorisation de mémoriser ou de stocker les informations de cette conversation aussi , 
        Vous êtes un assistant qui identifie le nom l'étudiant et identifie toutes  les questions , 
        les reponses associer et la ponderation (si ca existe) associer. 
        
        Si les reponses du manuscrit ne sont pas visibles ou un mot dans le manuscrit n'est pas visible, essayez de vous inspirer du
        contexte ou de message systeme si ce mot est là, cela pour ameliorer la detection des mots. N'ajouter pas des mots ou des textes qui n'existe pas
        si une reponse ou autres choses est vides, il faudrait laisser ca vide. utiliser le mesage systeme pour seulement pour les mots qui existes, mais invisibles,
        pour ameliorer la detection de mots.
               
        Si une réponse inclut une image, analysez son contenu visuel et fournissez une description concise et pertinente de l'image.
        
        Dire obligatoirement s'il y a des questions à choix multiples ou des questions de correspondance ou autres types des questions.
        
        Si le texte dans l'image ne peut pas être extrait directement, décrivez les éléments visuels présents, mais pas en detaille.
        
        {vision_prompt}
        
        """
        
        response = openai.ChatCompletion.create(
            model="chatgpt-4o-latest",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant temporaire, Vous êtes un assistant temporaire, Vous n'avez pas l'autorisation de mémoriser ou de stocker les informations de cette conversation, aussi ces copies de reference est utiliser pour aider à detecter de maniere efficace les textes invisibles sur les copies des etudiants (n'ajouter pas les mots et les textes qui semnlent n'est pas exister)"+reference_content},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            },
                        },
                    ],
                }
            ],
            
        )
        content = response['choices'][0]['message']['content']
        return content
    except openai.error.OpenAIError as e:
        st.error(f"Erreur lors de la communication avec l'API OpenAI: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur inattendue: {e}")
        return None

def grade_student_copy(reference_content, student_content, api_key, chatgpt_prompt,ortho_weight, syntax_weight, logic_weight): 
    try:
        openai.api_key = api_key
        
        prompt = f"""
        Réponse de référence :
        {reference_content}

        Réponse de l'étudiant :
        {student_content}

        Veuillez effectuer les tâches suivantes :
        {chatgpt_prompt}
        
        
        on doit avoir obligatoirement un Court Commentaire expliquant la note,
        en insistant sur les erreurs et les réussites si applicables(en expliquant chaque point attribuer à une question, justifier pourquoi à avoir donné des points à une questions )
        
        
        Formatez votre réponse comme suit :
        Nom : [nom de l'étudiant]
        Note : [0-100]
        Commentaire : [Commentaire]
        """



        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Vous êtes un assistant temporaire. Vous n'avez pas l'autorisation de mémoriser ou de stocker les informations de cette conversation aussi Vous êtes un assistant qui identifie les noms des étudiants et évalue leurs réponses en tenant compte des pondérations spécifiées pour chaque partie de la réponse de référence."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            top_p=1
        )

        content = response['choices'][0]['message']['content'].strip()
        lines = content.split('\n')

        # Initialisation des variables
        name = "Inconnu"
        score = "Erreur"
        feedback = "Pas de commentaire."

        # Parcourir les lignes pour extraire les informations
        for line in lines:
            if line.startswith("Nom"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Note"):
                score_text = line.split(":", 1)[1].strip()
                # Utiliser la méthode améliorée pour extraire la note
                score_match = re.search(r"([\d.,]+)(?:/([\d.,]+))?", score_text)
                if score_match:
                    numerator = float(score_match.group(1).replace(',', '.'))
                    denominator = score_match.group(2)
                    if denominator:
                        denominator = float(denominator.replace(',', '.'))
                        score = (numerator / denominator) * 100  # Normaliser si nécessaire
                    else:
                        score = numerator
                else:
                    score = "Erreur"
            elif line.startswith("Commentaire"):
                feedback = line.split(":", 1)[1].strip()

    except openai.error.OpenAIError as e:
        st.error(f"Erreur lors de la communication avec l'API OpenAI: {e}")
        name, score, feedback, content = "Erreur API", "Erreur", "Impossible d'évaluer.", "Content"

    except Exception as e:
        st.error(f"Erreur inattendue: {e}")
        name, score, feedback, content = "Erreur", "Erreur", "Erreur inattendue.", "Content"

    return name, score, feedback , content 

def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False)
    return output.getvalue().decode('utf-8')

def extract_images_from_pdf(pdf_file):
    try:
        images = []
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                image = Image.open(io.BytesIO(image_bytes))
                images.append(image)
        
        return images
    except Exception as e:
        st.error(f"Erreur lors de l'extraction des images du PDF: {e}")
        return []

def extract_latex_and_text(content):
    pattern = r'(?s)(.*?)\\\[([\s\S]*?)\\\]'

    matches = re.finditer(pattern, content)

    parts = []
    last_pos = 0

    for match in matches:
        before_latex = match.group(1)
        latex_content = match.group(2)
        
        if before_latex.strip():
            parts.append(('text', before_latex.strip()))

        parts.append(('latex', latex_content.strip()))
        last_pos = match.end()

    remaining_text = content[last_pos:].strip()
    if remaining_text:
        parts.append(('text', remaining_text))

    return parts

st.title("Système de correction des Copies")

st.header("Télécharger la Copie de Référence (PDF)")
reference_file = st.file_uploader("Téléchargez le PDF de la copie de référence", type=["pdf"])

st.header("Télécharger les Copies des Étudiants (PDFs)")
student_files = st.file_uploader("Téléchargez les PDFs des copies des étudiants", type=["pdf"], accept_multiple_files=True)

st.header("Définir les Niveaux de Correction")
ortho_weight = 30
syntax_weight =  40
logic_weight = 30

vision_prompt = st.text_area("Entrez le prompt pour la vision :", height=100)
chatgpt_prompt = st.text_area("Entrez le prompt pour la correction:", height=200)

api_key = st.secrets["API_KEY"]

if st.button("Lancer la correction"):
    if reference_file and student_files and api_key:
        with st.spinner('Correction en cours...'):
            try:
                reference_images = extract_images_from_pdf(reference_file)
                
                reference_texts = []
                for ref_img in reference_images:
                    ref_image_bytes = io.BytesIO()
                    ref_img.save(ref_image_bytes, format="PNG")
                    ref_image_base64 = base64.b64encode(ref_image_bytes.getvalue()).decode('utf-8')
                    reference_content = extract_content_from_image_reference(ref_image_base64, api_key, vision_prompt)
                    if reference_content:
                        reference_texts.append(reference_content)
                
                if not reference_texts:
                    st.warning("Impossible d'extraire du contenu de la copie de référence.")
                    st.stop()

                reference_text_combined = "\n".join(reference_texts)
                st.subheader("Copie de Référence")
                for ref_img in reference_images:
                    st.image(ref_img, caption="Copie de Référence")
                
                st.write("Contenu Extrait de la Copie de Référence :")
                parts = extract_latex_and_text(reference_text_combined)
                for part_type, content in parts:
                    if part_type == 'text':
                        st.write(content)
                    elif part_type == 'latex':
                        st.latex(content)
                
                results = []

                for student_file in student_files:
                    student_images = extract_images_from_pdf(student_file)
                    
                    student_texts = []
                    for student_img in student_images:
                        student_image_bytes = io.BytesIO()
                        student_img.save(student_image_bytes, format="PNG")
                        student_image_base64 = base64.b64encode(student_image_bytes.getvalue()).decode('utf-8')
                        student_content = extract_content_from_image(student_image_base64, api_key, vision_prompt,reference_content)
                        if student_content:
                            student_texts.append(student_content)
                    
                    if not student_texts:
                        st.warning(f"Impossible d'extraire du contenu pour la copie {student_file.name}.")
                        continue

                    student_text_combined = "\n".join(student_texts)
                    
                    
                    
                    st.subheader(f"Copie d'Étudiant : {student_file.name}")
                    for student_img in student_images:
                        st.image(student_img, caption=f"Copie d'Étudiant : {student_file.name}")
                    
                    st.write("Contenu Extrait de la Copie d'Étudiant :")
                    parts = extract_latex_and_text(student_text_combined)
                    for part_type, content in parts:
                        if part_type == 'text':
                            st.write(content)
                        elif part_type == 'latex':
                            st.latex(content)
                    
                    name, score, feedback, Contents = grade_student_copy(reference_text_combined, student_text_combined, api_key, chatgpt_prompt, ortho_weight, syntax_weight, logic_weight)
                    st.write(f"Nom: {name}, Note: {score}, Commentaire: {feedback}")
                    results.append({"Nom": name, "Note": score, "Commentaire": Contents})
                
                st.write(Contents)
                df_results = pd.DataFrame(results)
                st.dataframe(df_results)
                csv_data = to_csv(df_results)
                st.download_button("Télécharger les résultats au format CSV", data=csv_data, file_name="results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Une erreur s'est produite lors de la correction : {e}")
    else:
        st.warning("Veuillez télécharger tous les fichiers nécessaires et fournir la clé API.")


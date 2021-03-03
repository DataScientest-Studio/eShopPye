import streamlit as st
import awesome_streamlit as ast

import pages.presentation.presentation
import pages.donnees.donnees
import pages.classification_images.classification_images
import pages.classification_texte.classification_texte
import pages.classification_bimodale.classification_bimodale
import pages.conclusion.conclusion


# Load services
ast.core.services.other.set_logging_format()


# Dictionary of pages modules
PAGES = {
    "Présentation": pages.presentation.presentation,
    "Données": pages.donnees.donnees,
    "Classification d'image": pages.classification_images.classification_images,
    "Classification de texte": pages.classification_texte.classification_texte,
    "Classification bimodale": pages.classification_bimodale.classification_bimodale,
    "Conlusion": pages.conclusion.conclusion
}



def main():
    """Main function of the App"""
    
    # Create sidebar and set title
    st.sidebar.title("Menu")
    
    # Page selection
    selection = st.sidebar.radio("Aller à:", list(PAGES.keys()))
    page = PAGES[selection]
    
    # Show loading
    with st.spinner(f"Chargement de {selection}..."):
        ast.shared.components.write_page(page)



if __name__ == "__main__":
    main()
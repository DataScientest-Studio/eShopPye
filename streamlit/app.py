import streamlit as st
import awesome_streamlit as ast
import pages.classification_images

# Load services
ast.core.services.other.set_logging_format()

PAGES = {
    "Présentation": None,
    "Données": None,
    "Classification d'image": pages.classification_images,
    "Classification de texte": None,
    "Classification bimodale": None,
    "Conlusion": None
}



def main():
    """Main function of the App"""
    
    st.sidebar.title("Menu")
    
    # Page selection
    selection = st.sidebar.radio("Aller à:", list(PAGES.keys()))
    page = PAGES[selection]
    
    # Show loading
    with st.spinner(f"Chargement de {selection}..."):
        ast.shared.components.write_page(page)



if __name__ == "__main__":
    main()
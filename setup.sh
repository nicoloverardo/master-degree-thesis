mkdir -p ~/.streamlit/

printf "\
[general]\n\
email = \"n.verardo@outlook.com\"\n\
" > ~/.streamlit/credentials.toml

printf "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
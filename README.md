# DEEPSEEK LAW CHATBOT - Machine Learning 

## Overview
This project aims to provide the AI Assistant who can help learners explore and understand law concepts, drawing knowledge from the law on cyberinformation security 2015 and the law on cybersecurity 2018.


## Features

- **[Domain-specific Q&A]:** Chatbot can accurately answers law-related questions by processing data from the law.
- **[User-friendly UI]:** Intuitive user interface is design using Chainlit, enabling smooth interaction with the chatbot.
- **[Memory-enhanced interaction]:** Chatbot is equiped with the ability to retain conversational context, allowing responses that reflect prior interactions.
- **[Local Deployment]:** Chatbot are deployed locally, providing users with a stable, accessible experience without needing internet connectivity or external servers, ensuring responsiveness on standard devices.

## Installation 

1. Clone the repository
2. Navigate to the project directory
3. Install DeepSeek models via Ollama: ollama pull deepseek-r1: 1.5b
4. Install the required dependencies in the [requirements](https://github.com/ThaiDuongLe20022003/Intern-and-Thesis/blob/main/requirements.txt)
5. Adding your Chainlit auth secret, and Literal AI API in the .env file.You can find the instructions in the [Useful Links](#useful-links)

## Useful Links
- [Building Chatbot Instruction](https://docs.llamaindex.ai/en/stable/understanding/putting_it_all_together/chatbots/building_a_chatbot/)
- [Chainlit Document](https://docs.chainlit.io/get-started/overview)

## How To Run The Project
1. python data_loader.py
2. chainlit run app.py

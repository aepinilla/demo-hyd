# Troubleshooting Guide

This document provides solutions to common issues you might encounter when using the chatbot application.

## Common Issues and Solutions

### OpenAI API Key Issues

**Issue**: Error message about invalid API key or authentication.

**Solution**:
1. Check that your `.env` file contains the correct OpenAI API key.
2. Ensure the API key is properly formatted without extra spaces.
3. Verify that your OpenAI account is active and has available credits.

### Streamlit Interface Issues

**Issue**: Streamlit interface doesn't load or displays errors.

**Solution**:
1. Make sure Streamlit is properly installed: `pip install streamlit`.
2. Check if another Streamlit app is running on the same port.
3. Try restarting the application with `streamlit run app_new.py`.

### Import Errors

**Issue**: Python import errors when running the application.

**Solution**:
1. Make sure you're running the application from the project root directory.
2. Verify that all dependencies are installed: `uv sync`.
3. Check that the project structure is intact with all necessary files.

### Slow Response Times

**Issue**: The chatbot takes a long time to respond.

**Solution**:
1. Check your internet connection.
2. The OpenAI API might be experiencing high traffic.
3. Consider using a faster model in `src/config/settings.py`.

## Debugging Tips

### Enable Verbose Logging

Add this code at the top of your script to enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Streamlit Session State

To debug issues with session state, add this code to your app:

```python
st.write("Session State:", st.session_state)
```

### Inspect API Responses

To see the raw responses from the OpenAI API, modify the `get_openai_response` function in `src/core/llm.py` to print the response before returning it.

## Getting Help

If you continue to experience issues:

1. Check the [Streamlit documentation](https://docs.streamlit.io/).
2. Review the [OpenAI API documentation](https://platform.openai.com/docs/api-reference).
3. Search for similar issues on Stack Overflow or GitHub.

import os
import chainlit as cl 
from sage import app
from langchain.schema.runnable import RunnableConfig


welcome_message = "Welcome! Ask anything about your stored documents and get AI-powered insights in seconds."

@cl.on_chat_start  
async def start_chat():
    await cl.Message(content=welcome_message).send()
    cl.user_session.set("runnable", app)


@cl.on_message 
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    input = {"question": message.content}

    value = None
    for output in runnable.stream(input):
        for key, value in output.items():
            print(f"Finished running: {key}:")
            if key == "generator_agent":
                answer = value["answer"]
                await msg.stream_token(answer)

    await msg.send()
import os
import logging
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import nst

logging.basicConfig(level=logging.INFO)

bot_token = os.environ.get('TG_BOT_TOKEN')
if not bot_token:
    raise ValueError('TG_BOT_TOKEN environment variable is not set.')

bot = Bot(token=bot_token)
dp = Dispatcher(bot, storage=MemoryStorage())

class Form(StatesGroup):
    UploadContent = State()
    UploadStyle = State()

@dp.message_handler(commands='start')
async def cmd_start(message: types.Message):
    await bot.send_photo(message.chat.id, open('images_for_bot/start.jpg', 'rb'))

    await bot.send_message(message.chat.id, f'Hello, {message.from_user.first_name}!\n'
                                             f'I am a representative of the art studio and I can magically change the style of your image 🎨')

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button = types.KeyboardButton('Upload image for transformation')
    markup.add(button)

    await bot.send_message(message.chat.id, 'Please click the button to upload an image for style transformation.', reply_markup=markup)

@dp.message_handler(lambda message: message.text == 'Upload image for transformation')
async def process_upload_photo(message: types.Message):
    await bot.send_message(message.chat.id, 'Please upload the content image.')

    await Form.UploadContent.set()

@dp.message_handler(commands='transfer_style')
async def cmd_transfer_style(message: types.Message):
    await bot.send_message(message.chat.id, 'Please upload the content image.')

    await Form.UploadContent.set()

@dp.message_handler(content_types=['photo'], state=Form.UploadContent)
async def process_content_photo(message: types.Message, state: FSMContext):
    await message.photo[-1].download('images/content_image.jpg')
    await bot.send_message(message.chat.id, 'Content image successfully uploaded!')

    await bot.send_message(message.chat.id, 'Now, please upload the style image.')
    await Form.next()

@dp.message_handler(content_types=['photo'], state=Form.UploadStyle)
async def process_style_photo(message: types.Message, state: FSMContext):
    await message.photo[-1].download('images/style_image.jpg')
    await bot.send_message(message.chat.id, 'Style image successfully uploaded! Starting image processing. This will only take a moment!) 🔮️')
    await launch_nst(message)
    await state.finish()

async def launch_nst(message):
    content_image_name = 'images/content_image.jpg'
    style_image_name = 'images/style_image.jpg'

    await bot.send_message(message.chat.id, 'Processing the images...', reply_markup=None)
    with open('images_for_bot/working.jpg', 'rb') as working_image:
        await bot.send_photo(message.chat.id, working_image)

    nst.main(content_image_name, style_image_name)

    await bot.send_message(message.chat.id, 'Done!')

    with open('images/bot-result.png', 'rb') as result:
        await bot.send_photo(message.chat.id, result)

    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    button = types.KeyboardButton('Upload image for transformation')
    markup.add(button)

    await bot.send_message(message.chat.id, 'To try again, simply click the button below to upload a new image for style transformation. 💫',
                           reply_markup=markup)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)

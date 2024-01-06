from PIL import Image, UnidentifiedImageError
import os

def resize_and_change_color(input_folder, output_folder, target_size,colors_number):
    # Sprawdź, czy folder wyjściowy istnieje, jeśli nie, utwórz go
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Przejście przez podfoldery w folderze wejściowym
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            # Sprawdź, czy plik jest obrazem
            if filename.lower().endswith(('.png')):
                input_path = os.path.join(root, filename)

                try:
                    # Wczytaj obraz
                    image = Image.open(input_path)

                    # Zmiana rozmiaru
                    resized_image = image.resize(target_size)

                    # Zmiana kolorów
                    colored_image = resized_image.convert('RGBA')
                    colored_image = colored_image.convert('P', palette=Image.ADAPTIVE, colors=colors_number)

                    # # Utwórz ścieżkę do pliku wynikowego bez zachowania struktury folderow
                    # folder_name = os.path.basename(os.path.normpath(root))  # Nazwa folderu z ktorego pochodzi img
                    # output_filename = f"{folder_name}_{filename}"  # Nowa nazwa pliku
                    # output_path = os.path.join(output_folder, output_filename)

                    # # zachowaj strukture folderow
                    relative_path = os.path.relpath(input_path, input_folder)
                    output_path = os.path.join(output_folder, relative_path)
                    # # Utwórz katalog, jeśli nie istnieje
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Zapisz zmodyfikowany obraz
                    colored_image.save(output_path)
                except (UnidentifiedImageError, Exception, OSError) as e:
                    print(f"Nie można zidentyfikować pliku obrazu: {input_path}, błąd: {e}")

colors_number = 20
input_folder = "..\scrapperEmote"
output_folder = "..\projekt\img"
target_size = (64,64)  # Nowy rozmiar obrazu

# Wywołaj funkcję
resize_and_change_color(input_folder, output_folder, target_size, colors_number)

print("\n\nKONIEC")

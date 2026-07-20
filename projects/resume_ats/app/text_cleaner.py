import re
import unicodedata


class TextCleaner:


    def clean_text(self, text):

        """
        Keeps:
        - alphabets
        - numbers
        - spaces
        - new lines
        - @
        - .
        - +
        - -

        Removes:
        - unicode symbols
        - emojis
        - brackets
        - unwanted characters
        """


        if not isinstance(text, str):

            return ""


        # Unicode normalize

        text = unicodedata.normalize(

            "NFKD",

            text

        )



        # Remove unicode symbols/emojis

        text = "".join(

            ch

            for ch in text

            if not unicodedata.category(ch).startswith("S")

        )



        # Keep required characters

        text = re.sub(

            r"[^a-zA-Z0-9\s@.\+\-\n]",

            "",

            text

        )



        # Remove extra spaces

        text = re.sub(

            r"[ \t]+",

            " ",

            text

        )



        # Remove empty lines

        lines = []


        for line in text.splitlines():

            line = line.strip()


            if line:

                lines.append(line)



        return "\n".join(lines)
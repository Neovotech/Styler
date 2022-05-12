# Styler

After cloning the codebase:
please clone the code base for stylegan3 and CLIP -> 

git clone https://github.com/NVlabs/stylegan3


git clone https://github.com/openai/CLIP


# Initial Questions

Ever found a jacket or sweater with a great fit that’s comfortable to wear? Alas, it’s plain black and boring. 

Wondering what it might look like otherwise, in a different variation - another colour, pattern, size, or worn on a different person entirely.

What about being a designer? You might be stumped and find it hard to come up with a creative new design or pattern.

# Swagger API
API endpoint:
http://34.255.121.73:8081/docs


POST Method Name: /generate/

example JSON format:

    {
        "gender": "male",
        "apparel_type": "tshirt",
        "colour": "blue",
        "characteristics": "",
        "placing": "upper",
        "number_of_clothes": 9
    }

![API Endpoint](/images/00.png)


# Results

Here are some of the generated results.

`"Long Sleeve Grey Shirt"`
![Results 1](/images/1.png)

`"Yellow Floral Dress"`
![Results 2](/images/2.png)

`"Female Long Sleeve Plaid Shirt"`
![Results 3](/images/3.png)

`"Blue Sports Jacket"`
![Results 4](/images/4.png)

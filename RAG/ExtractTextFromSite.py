from selenium import webdriver
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import re

# # Setup Chrome driver
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import time

# def expand_all(driver):
#     expanded = set()
#     while True:
#         buttons = driver.find_elements(By.CLASS_NAME, "expand-collapse")
#         new_clicks = 0
#         for btn in buttons:
#             parent = btn.find_element(By.XPATH, "..")
#             node_id = parent.get_attribute("data-id")
#             if node_id and node_id not in expanded:
#                 try:
#                     driver.execute_script("arguments[0].scrollIntoView(true);", btn)
#                     btn.click()
#                     expanded.add(node_id)
#                     time.sleep(0.3)
#                     new_clicks += 1
#                 except:
#                     pass
#         if new_clicks == 0:
#             break

# # Setup
# options = Options()
# options.add_argument("--start-maximized")
# driver = webdriver.Chrome(options=options)
# driver.get("https://help.autodesk.com/view/RVT/2025/ENU/")
# time.sleep(5)

# expand_all(driver)  # Recursively expand all nodes

# # Get all links
# links = driver.find_elements(By.CSS_SELECTOR, "li.node-tree-item a[data-url]")

# print(f"\nTotal links found: {len(links)}\n")
# for link in links:
#     url = link.get_attribute("data-url")
#     if url:
#         print("https://help.autodesk.com" + url)

# with open("revit_links.txt", "w", encoding="utf-8") as file:
#     for link in links:
#         url_for_file = link.get_attribute("data-url")
#         file.write("https://help.autodesk.com" + url_for_file + "\n")

# driver.quit()




        # Check if the link matches the desired pattern
        # if re.search(r"guid=Revit_API_Revit_API_Developers_Guide_Basic_Interaction_with_Revit_Elements_Filtering", href):
        #     print(href)
        #     # Click the link
        #     link.click()
        #     time.sleep(3)
            
        #     # Extract content from the page
        #     soup = BeautifulSoup(driver.page_source, 'html.parser')
        #     text = soup.get_text(separator="\n", strip=True)
        #     print(text)
            
        #     # Go back to the previous page
        #     driver.back()
        #     time.sleep(3)
    # except Exception as e:
    #     print(f"Error clicking element: {e}")


# # Base URL of the documentation
# base_url = "https://help.autodesk.com/view/RVT/2025/ENU/"

# # Navigate to the main Filtering page
# driver.get(base_url + "?guid=Revit_API_Revit_API_Developers_Guide_Basic_Interaction_with_Revit_Elements_Filtering_html")
# time.sleep(3)

# # Extract TOC links from the page
# toc_links = []
# try:
#     # Locate the TOC container; adjust the selector based on actual HTML structure
#     toc_container = driver.find_element(By.CSS_SELECTOR, "div.toc")  # Example selector
#     links = toc_container.find_elements(By.TAG_NAME, "a")
#     for link in links:
#         href = link.get_attribute("href")
        
#         if href:
#             toc_links.append(href)
# except Exception as e:
#     print(f"Error extracting TOC links: {e}")

# # Remove duplicate links
# toc_links = list(set(toc_links))
driver = webdriver.Chrome()
                
toc_links = []
# Read the file line by line
with open("revit_links.txt", 'r', encoding='utf-8') as file:
    for line in file:
        # Strip whitespace and check if the line is a valid link
        line = line.strip()
        if line and line.startswith('http'):
            toc_links.append(line)

# Extract content from each linked page
all_text = ""
for i, link in enumerate(toc_links):
    try:
        driver.get(link)
        time.sleep(2)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        text = soup.get_text(separator="\n", strip=True)
        all_text += f"\n\n--- Page {i+1}: {link} ---\n{text}"
    except Exception as e:
        print(f"Failed to process {link}: {e}")

driver.quit()

# Save the extracted content to a file
with open("autodesk_docs_all.txt", "w", encoding="utf-8") as file:
    file.write(all_text)


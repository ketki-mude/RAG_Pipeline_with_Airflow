from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select, WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import requests
from s3_utils import upload_file_to_s3

def fetch_nvidia_financial_reports():
    url = "https://investor.nvidia.com/financial-info/quarterly-results/default.aspx"
    
    # Set up Selenium WebDriver options
    options = webdriver.ChromeOptions()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36")

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    try:
        print(f"Navigating to URL: {url}")
        driver.get(url)

        wait = WebDriverWait(driver, 10)
        dropdown_element = wait.until(
            EC.presence_of_element_located((By.ID, "_ctrl0_ctl75_selectEvergreenFinancialAccordionYear"))
        )

        select = Select(dropdown_element)
        years = [option.text for option in select.options if int(option.text) >= 2021]
        print(years)
        reports = []  # List to store fetched reports

        for year in years:
            print(f"\nProcessing year: {year}")
            select.select_by_visible_text(year)
            time.sleep(2)  

            quarters = driver.find_elements(By.XPATH, "//div[contains(@class, 'evergreen-accordion-header')]")
            print(f"Found {len(quarters)//2} quarters for year {year}. Processing...")

            for quarter in quarters:
                try:
                    quarter_heading = quarter.text.strip()

                    minus_icon_present = quarter.find_elements(By.XPATH, ".//span[contains(@class, 'accordion-toggle-icon evergreen-icon-minus')]")
                    if not minus_icon_present:
                        plus_icons = quarter.find_elements(By.XPATH, ".//span[contains(@class, 'accordion-toggle-icon evergreen-icon-plus')]")
                        if plus_icons:  
                            print(f"Expanding quarter '{quarter_heading}'...")
                            driver.execute_script("arguments[0].scrollIntoView(true);", plus_icons[0])
                            plus_icons[0].click()
                            time.sleep(2)  

                    expanded_content_elements = quarter.find_elements(By.XPATH, "../div[contains(@class, 'evergreen-accordion-content')]")

                    if expanded_content_elements:
                        links = expanded_content_elements[0].find_elements(By.XPATH, ".//a")

                        for link in links:
                            link_text = link.text.strip()
                            if "10-K" in link_text or "10-Q" in link_text:
                                pdf_url = link.get_attribute('href')
                                if pdf_url:
                                    print(f"Fetching PDF for '{quarter_heading}'...")

                                    response = requests.get(pdf_url, stream=True)
                                    pdf_filename = f"{year}_{ '_'.join(quarter_heading.split()[:2])}.pdf"
                                    s3_key = f"documents/pdf/{year}/{pdf_filename}"
                                    s3_url = upload_file_to_s3(response.content, s3_key)
                                    reports.append({
                                        "pdf_filename": pdf_filename,
                                        "content": response.content,
                                        "s3_url": s3_url,
                                        "year": year
                                    })
                                    print(f"Uploaded PDF to S3: {pdf_filename}")

                except Exception as e:
                    #print(f"Uploaded PDF to S3: {status}")
                    print(f"Error processing quarter '{quarter_heading}': {e}")

        return reports

    finally:
        print("\nClosing WebDriver...")
        driver.quit()

# Example usage
# reports = fetch_nvidia_financial_reports()
# for report in reports:
#     print(f"Fetched: {report['pdf_filename']} (Size: {len(report['content'])} bytes)")

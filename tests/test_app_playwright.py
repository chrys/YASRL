import pytest
from playwright.sync_api import Page, expect
import time
import psycopg2
import os

BASE_URL = "http://localhost:5000"

@pytest.fixture
def db_conn():
    # Use your actual environment variable or connection string
    conn = psycopg2.connect(os.environ["POSTGRES_URI"])
    yield conn
    conn.close()

def delete_project_by_name(conn, name):
    with conn.cursor() as cursor:
        cursor.execute("DELETE FROM projects WHERE name = %s", (name,))
        conn.commit()
        
def delete_qa_pair_by_question(conn, project_id, source, question):
    with conn.cursor() as cursor:
        cursor.execute(
            "DELETE FROM project_qa_pairs WHERE project_id = %s AND source = %s AND question = %s",
            (project_id, source, question)
        )
        conn.commit()

def test_create_project(page: Page, db_conn):
    unique_name = f"Third Project {int(time.time())}"
    page.goto(BASE_URL)
    page.get_by_role("link", name="Admin").click()
    page.get_by_role("textbox", name="Project Name *").click()
    page.get_by_role("textbox", name="Project Name *").fill(unique_name)
    page.get_by_role("textbox", name="Project Name *").press("Tab")
    page.get_by_role("textbox", name="Description").fill("third description")
    page.get_by_role("button", name="Create Project").click()
    expect(page.get_by_text(unique_name)).to_be_visible()
    # Cleanup: remove the project from the database
    delete_project_by_name(db_conn, unique_name)
    
def test_select_project(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("link", name="Admin").click()
    # Wait for either a project or the empty state
    project_items = page.locator(".project-item")
    page.wait_for_timeout(500)  # Give time for HTMX to load projects (adjust as needed)
    if project_items.count() == 0:
        # No projects found, check for the empty state message
        expect(page.locator("text=No projects found.")).to_be_visible()
    else:
        # Select the first project
        first_project = project_items.first
        project_name = first_project.locator(".project-name").inner_text()
        first_project.click()
        # Assert project details card appears (project name should be visible)
        expect(page.get_by_role("heading", name=project_name)).to_be_visible()
        # Assert Sources section appears
        expect(page.get_by_role("heading", name="Sources", exact=True)).to_be_visible()
        # Assert "Add Sources" section appears
        expect(page.get_by_role("heading", name="Add Sources")).to_be_visible()
    
    
def test_chat_page(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("link", name="Chat").click()
    # Wait for the dropdown to be present
    project_select = page.locator("#project-select")
    expect(project_select).to_be_visible()
    # Wait for options to be populated (other than the placeholder)
    page.wait_for_timeout(500)  # Adjust if needed for HTMX/JS load

    # Get all option texts except the placeholder
    options = project_select.locator("option").all()
    option_texts = [opt.inner_text().strip() for opt in options if opt.inner_text().strip() != "-- Select a project --"]

    # Assert that at least one project is present
    assert len(option_texts) > 0, "No projects found in chat dropdown"


def test_evaluate_page_project_selection(page: Page):
    page.goto(BASE_URL)
    page.get_by_role("link", name="Evaluate").click()
    # Wait for the project dropdown to be present
    project_select = page.locator("#eval-project-select")
    expect(project_select).to_be_visible()
    page.wait_for_timeout(500)  # Adjust if needed for HTMX/JS load

    # Get all option elements except the placeholder
    options = project_select.locator("option").all()
    real_options = [opt for opt in options if opt.inner_text().strip() != "-- Select a project --"]

    # Assert that at least one project is present
    assert len(real_options) > 0, "No projects found in evaluate dropdown"

    # Select the first project
    first_option_value = real_options[0].get_attribute("value")
    project_select.select_option(first_option_value)
    page.wait_for_timeout(500)  # Wait for sections to load

    # Assert that the "Select Source" heading appears (twice if needed)
    expect(page.get_by_role("heading", name="Select Source")).to_be_visible()
    # If you expect two such headings, you can check count:
    assert page.get_by_role("heading", name="Select Source").count() >= 1


def test_evaluate_page_new_qa_pair(page: Page, db_conn):
    page.goto(BASE_URL)
    page.get_by_role("link", name="Evaluate").click()

    # Select the first project
    project_select = page.locator("#eval-project-select")
    expect(project_select).to_be_visible()
    page.wait_for_timeout(500)
    options = project_select.locator("option").all()
    real_options = [opt for opt in options if opt.inner_text().strip() != "-- Select a project --"]
    assert len(real_options) > 0, "No projects found in evaluate dropdown"
    first_option = real_options[0]
    first_option_value = first_option.get_attribute("value")
    project_select.select_option(first_option_value)
    page.wait_for_timeout(500)

    # Select the first source
    source_select = page.locator("#eval-source-select")
    expect(source_select).to_be_visible()
    page.wait_for_timeout(500)
    source_options = source_select.locator("option").all()
    real_source_options = [opt for opt in source_options if opt.inner_text().strip() != "-- Select a source --"]
    assert len(real_source_options) > 0, "No sources found for selected project"
    first_source = real_source_options[0]
    first_source_value = first_source.get_attribute("value")
    source_select.select_option(first_source_value)
    page.wait_for_timeout(500)

    # Fill in the QA pair form
    question = "question 1"
    answer = "answer 1"
    context = "context 1"
    page.get_by_role("textbox", name="Question *").fill(question)
    page.get_by_role("textbox", name="Answer *").fill(answer)
    page.get_by_role("textbox", name="Context (Optional)").fill(context)

    # Handle alert dialog if it appears
    page.once("dialog", lambda dialog: dialog.dismiss())

    # Click the Add QA Pair button
    page.get_by_role("button", name="Add QA Pair").click()
    page.wait_for_timeout(1000)  # Wait for QA pairs to reload

    # Assert the new QA pair appears in the list
    expect(page.get_by_text("Q: question")).to_be_visible()
    expect(page.get_by_text("A: answer")).to_be_visible()
    expect(page.get_by_text("Context: context")).to_be_visible()

    # --- Cleanup: remove the QA pair from the database ---
    # Get project_id and source from the selected options
    assert first_option_value is not None, "Project option value should not be None"
    assert first_source_value is not None, "Source option value should not be None"
    project_id = int(first_option_value)
    source = first_source_value
    delete_qa_pair_by_question(db_conn, project_id, source, question)
    
    
# query_with_filters.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import RAGPipeline
import json
import logging

# Reduce logging noise
logging.basicConfig(level=logging.WARNING)

def interactive_with_filters():
    print("\n" + "="*60)
    print("🔍 RAG Query with Metadata Filters")
    print("="*60)
    print("📋 Available Commands:")
    print("  /skills python,java     - Filter by skills")
    print("  /section experience     - Filter by section type")
    print("  /page 2                - Filter by page number")
    print("  /clear                  - Clear all filters")
    print("  /help                   - Show this menu")
    print("  /show                   - Show current filters")
    print("  /debug                  - Show debug info")
    print("  quit                    - Exit\n")
    
    rag = RAGPipeline()
    current_filters = {}
    
    while True:
        try:
            query = input("❓ You: ").strip()
            
            # Check for exit
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                continue
            
            # Parse commands
            if query.startswith('/'):
                parts = query.split(' ', 1)
                command = parts[0].lower()
                
                # Handle commands WITHOUT values
                if command == '/help':
                    print("\n📋 Available Commands:")
                    print("  /skills python,java     - Filter by skills")
                    print("  /section experience     - Filter by section type")
                    print("  /page 2                - Filter by page number")
                    print("  /clear                  - Clear all filters")
                    print("  /help                   - Show this menu")
                    print("  /show                   - Show current filters")
                    print("  /debug                  - Show debug info")
                    print("  quit                    - Exit\n")
                    continue
                
                elif command == '/clear':
                    current_filters = {}
                    print("✅ All filters cleared!")
                    continue
                
                elif command == '/show':
                    if current_filters:
                        print(f"\n🔍 Active filters: {json.dumps(current_filters, indent=2)}\n")
                    else:
                        print("\nℹ️ No active filters\n")
                    continue
                
                elif command == '/debug':
                    print("\n🐛 Debug Info:")
                    print(f"  Filters: {json.dumps(current_filters)}")
                    print(f"  Index: {rag.index}")
                    print(f"  Model: {rag.model}")
                    continue
                
                # Handle commands WITH values
                if len(parts) != 2:
                    print("\n❌ Usage: /command value")
                    print("Example: /skills python,java")
                    print("Type /help for available commands\n")
                    continue
                
                value = parts[1].strip()
                actual_query = None
                
                # Process filter commands
                if command == '/skills':
                    skill_list = [s.strip() for s in value.split(',') if s.strip()]
                    if skill_list:
                        current_filters['detected_skills'] = skill_list
                        print(f"✅ Filtering by skills: {', '.join(skill_list)}")
                        actual_query = f"Tell me about skills including {', '.join(skill_list)}"
                    else:
                        print("❌ No valid skills provided. Use: /skills python,java")
                        continue
                
                elif command == '/section':
                    valid_sections = ['experience', 'education', 'skills', 'projects', 'summary', 'general']
                    if value.lower() in valid_sections:
                        current_filters['section_type'] = value.lower()
                        print(f"✅ Filtering by section: {value}")
                        actual_query = f"What's in the {value} section?"
                    else:
                        print(f"❌ Invalid section. Valid: {', '.join(valid_sections)}")
                        continue
                
                elif command == '/page':
                    try:
                        page_num = int(value)
                        if page_num > 0:
                            current_filters['page_number'] = page_num
                            print(f"✅ Filtering by page: {page_num}")
                            actual_query = f"What's on page {page_num}?"
                        else:
                            print("❌ Page number must be positive")
                            continue
                    except ValueError:
                        print("❌ Invalid page number. Use: /page 1")
                        continue
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("Type /help for available commands")
                    continue
                
                # Execute the query
                if actual_query:
                    print(f"\n🔍 Searching with filters...")
                    result = rag.ask_with_filters(actual_query, current_filters)
                    
                    print("\n" + "="*60)
                    print(f"💡 Answer: {result['answer']}")
                    print("="*60)
                    
                    if result.get('sources'):
                        print(f"\n📚 Sources ({len(result['sources'])} chunks):")
                        for i, source in enumerate(result['sources'], 1):
                            skills = source.get('skills', [])[:3]
                            skills_str = f" [Skills: {', '.join(skills)}]" if skills else ""
                            print(f"  {i}. Page {source.get('page', 'unknown')} - {source.get('section', 'general')}{skills_str}")
                    
                    if result.get('error'):
                        print(f"\n⚠️ Error: {result['error']}")
                    
                    print("-"*60 + "\n")
                
                continue
            
            # Regular question (not a command)
            print(f"\n🔍 Searching... (Active filters: {json.dumps(current_filters) if current_filters else 'None'})")
            result = rag.ask_with_filters(query, current_filters)
            
            print("\n" + "="*60)
            print(f"💡 Answer: {result['answer']}")
            print("="*60)
            
            if result.get('sources'):
                print(f"\n📚 Sources ({len(result['sources'])} chunks):")
                for i, source in enumerate(result['sources'], 1):
                    skills = source.get('skills', [])[:3]
                    skills_str = f" [Skills: {', '.join(skills)}]" if skills else ""
                    print(f"  {i}. Page {source.get('page', 'unknown')} - {source.get('section', 'general')}{skills_str}")
            
            if result.get('error'):
                print(f"\n⚠️ Error: {result['error']}")
            
            if current_filters:
                print(f"\n🔍 Active filters: {json.dumps(current_filters, indent=2)}")
            print("-"*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("Please try again or type 'quit' to exit\n")

if __name__ == "__main__":
    interactive_with_filters()
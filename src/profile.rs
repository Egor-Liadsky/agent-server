//! Профиль владельца: явно заданный набор предпочтений (роль, стиль,
//! формат, ограничения), подставляемый в системное сообщение каждого
//! запроса чата (specs/user-profiles, design.md).
//!
//! Встроенные профили (`teacher`, `psychologist`, `reviewer`) — константы
//! кода, а не строки в БД (design.md, решение 3): это операторские
//! умолчания сервиса, как `DEFAULT_SYSTEM_PROMPT`, и семантика их правки не
//! должна требовать миграции данных.

use crate::state::AppState;
use crate::store;

const PROFILE_HEADING: &str = "Профиль:";

/// Профиль, готовый к применению — встроенный или собственный владельца,
/// приведённые к одному виду.
#[derive(Debug, Clone)]
pub struct Profile {
    pub id: String,
    pub name: String,
    pub persona: String,
    pub style: String,
    pub format: String,
    pub constraints: Vec<String>,
    pub built_in: bool,
}

/// Три встроенных профиля сервиса (specs/user-profiles, «Встроенные
/// профили»). Порядок фиксирован и используется списком.
pub fn built_in_profiles() -> Vec<Profile> {
    vec![
        Profile {
            id: "teacher".to_string(),
            name: "Преподаватель".to_string(),
            persona: "Ты — терпеливый преподаватель, который объясняет тему так, чтобы её понял новичок."
                .to_string(),
            style: "Простой язык, короткие предложения, примеры перед абстракцией.".to_string(),
            format: "Объяснение по шагам, в конце — краткое резюме.".to_string(),
            constraints: vec![
                "Не используй термин без объяснения при первом употреблении".to_string(),
                "Проверяй понимание встречным вопросом, если тема сложная".to_string(),
            ],
            built_in: true,
        },
        Profile {
            id: "psychologist".to_string(),
            name: "Психолог".to_string(),
            persona: "Ты — внимательный собеседник, поддерживающий и помогающий разобраться в чувствах и ситуации."
                .to_string(),
            style: "Мягкий, эмпатичный тон, без оценочных суждений и готовых диагнозов.".to_string(),
            format: "Сначала отражение услышанного, затем уточняющий вопрос или бережный совет.".to_string(),
            constraints: vec![
                "Не ставь медицинские или психиатрические диагнозы".to_string(),
                "При признаках острого кризиса — прямо порекомендуй обратиться к специалисту".to_string(),
            ],
            built_in: true,
        },
        Profile {
            id: "reviewer".to_string(),
            name: "Технический ревьюер".to_string(),
            persona: "Ты — строгий технический ревьюер кода, ищущий баги и слабые места решения.".to_string(),
            style: "Прямой, конкретный, без похвалы ради вежливости.".to_string(),
            format: "Список находок с указанием места и предлагаемым исправлением.".to_string(),
            constraints: vec![
                "Не переписывай решение целиком без запроса — только точечные замечания".to_string(),
                "Отделяй критичные находки от стилевых придирок".to_string(),
            ],
            built_in: true,
        },
    ]
}

pub fn find_built_in(id: &str) -> Option<Profile> {
    built_in_profiles().into_iter().find(|p| p.id == id)
}

fn from_owner_profile(owner_profile: store::OwnerProfile) -> Profile {
    Profile {
        id: owner_profile.id,
        name: owner_profile.name,
        persona: owner_profile.persona,
        style: owner_profile.style,
        format: owner_profile.format,
        constraints: owner_profile.constraints,
        built_in: false,
    }
}

/// Список профилей, доступных владельцу: встроенные, затем собственные
/// (specs/user-profiles, «Встроенные профили видны без создания»,
/// «Созданный профиль виден в списке»).
pub async fn list(state: &AppState, owner: &str) -> Result<Vec<Profile>, store::StoreError> {
    let mut profiles = built_in_profiles();
    let owned = store::list_owner_profiles(&state.db, owner).await?;
    profiles.extend(owned.into_iter().map(from_owner_profile));
    Ok(profiles)
}

/// Один профиль, доступный владельцу: сперва встроенные, затем собственные
/// хранилища. Чужой профиль неотличим от несуществующего
/// (specs/user-profiles, «Чужой профиль не читается»).
pub async fn find(state: &AppState, owner: &str, id: &str) -> Result<Option<Profile>, store::StoreError> {
    if let Some(profile) = find_built_in(id) {
        return Ok(Some(profile));
    }
    match store::load_owner_profile(&state.db, owner, id).await {
        Ok(owner_profile) => Ok(Some(from_owner_profile(owner_profile))),
        Err(store::StoreError::NotFound) => Ok(None),
        Err(err) => Err(err),
    }
}

/// Профиль без единого непустого поля предпочтений бессмысленен
/// (specs/user-profiles, «Профиль без предпочтений отклоняется»).
pub fn has_any_preference(persona: &str, style: &str, format: &str, constraints: &[String]) -> bool {
    !persona.trim().is_empty()
        || !style.trim().is_empty()
        || !format.trim().is_empty()
        || constraints.iter().any(|c| !c.trim().is_empty())
}

fn raw_section(profile: &Profile) -> String {
    let mut lines = vec![PROFILE_HEADING.to_string()];
    if !profile.persona.trim().is_empty() {
        lines.push(format!("Роль: {}", profile.persona));
    }
    if !profile.style.trim().is_empty() {
        lines.push(format!("Стиль: {}", profile.style));
    }
    if !profile.format.trim().is_empty() {
        lines.push(format!("Формат ответа: {}", profile.format));
    }
    let constraints: Vec<&String> = profile.constraints.iter().filter(|c| !c.trim().is_empty()).collect();
    if !constraints.is_empty() {
        lines.push("Ограничения:".to_string());
        for constraint in constraints {
            lines.push(format!("- {constraint}"));
        }
    }
    lines.join("\n")
}

/// Раздел профиля, собранный и, при необходимости, усечённый по
/// `AGENTD_PROFILE_MAX_CHARS` (specs/user-profiles, «Слишком длинный раздел
/// профиля усекается»).
pub struct BuiltSection {
    pub section: String,
    pub chars: u32,
    pub truncated: bool,
}

/// Действующий профиль запроса: настройка чата поверх операторского
/// умолчания (specs/user-profiles, «Профиль выбирается настройкой чата
/// поверх операторского умолчания»). Пусто и там, и там — запрос собирается
/// без раздела профиля.
pub fn effective_profile_id(state: &AppState, settings: &agentcore::config::ChatSettings) -> Option<String> {
    settings.profile_id.clone().or_else(|| state.config.default_profile.clone())
}

pub fn build_section(profile: &Profile, max_chars: u32) -> BuiltSection {
    let full = raw_section(profile);
    let full_chars = full.chars().count() as u32;
    if full_chars <= max_chars {
        return BuiltSection { section: full, chars: full_chars, truncated: false };
    }
    let truncated: String = full.chars().take(max_chars as usize).collect();
    let chars = truncated.chars().count() as u32;
    BuiltSection { section: truncated, chars, truncated: true }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_in_profiles_are_exactly_three_and_have_preferences() {
        let profiles = built_in_profiles();
        assert_eq!(profiles.len(), 3);
        for profile in &profiles {
            assert!(profile.built_in);
            assert!(has_any_preference(&profile.persona, &profile.style, &profile.format, &profile.constraints));
        }
    }

    #[test]
    fn section_has_named_subsections_and_skips_empty_ones() {
        let profile = Profile {
            id: "custom".to_string(),
            name: "Свой".to_string(),
            persona: String::new(),
            style: "кратко".to_string(),
            format: String::new(),
            constraints: vec!["без воды".to_string()],
            built_in: false,
        };
        let built = build_section(&profile, 10_000);
        assert!(!built.section.contains("Роль:"));
        assert!(built.section.contains("Стиль: кратко"));
        assert!(!built.section.contains("Формат ответа:"));
        assert!(built.section.contains("Ограничения:\n- без воды"));
        assert!(!built.truncated);
    }

    #[test]
    fn section_longer_than_limit_is_truncated_and_flagged() {
        let profile = Profile {
            id: "custom".to_string(),
            name: "Свой".to_string(),
            persona: "а".repeat(100),
            style: String::new(),
            format: String::new(),
            constraints: Vec::new(),
            built_in: false,
        };
        let built = build_section(&profile, 20);
        assert_eq!(built.chars, 20);
        assert!(built.truncated);
        assert_eq!(built.section.chars().count(), 20);
    }

    #[test]
    fn empty_preferences_are_rejected() {
        assert!(!has_any_preference("", "", "", &[]));
        assert!(!has_any_preference("  ", "", "", &["  ".to_string()]));
        assert!(has_any_preference("", "стиль", "", &[]));
    }
}
